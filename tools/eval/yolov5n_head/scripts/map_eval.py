#!/usr/bin/env python3
"""
End-to-end mAP comparison for yolov5n_head.

WHAT THIS DOES (in plain words):
  1.  Read 466 head images from valid set
  2.  For each image, run THREE different "brains":
        A. ONNX  — original model via onnxruntime (this is "ground truth")
        B. FP16  — RKNN fp16 via simulator (quantized but still 16-bit)
        C. INT8  — RKNN int8 via simulator (the most compressed)
  3.  Each brain outputs raw numbers (18 channels x H x W)
  4.  Apply IDENTICAL post-processing to all three (letterbox reverse, NMS, conf threshold)
  5.  Compare each brain's detections against the human-labeled ground truth boxes
  6.  Compute mAP@0.5 and mAP@0.5:0.95 for each brain
  7.  Print summary table to console + log file

USAGE:
  cd /home/developer/camera-auto-tracking/ArcFoundry.eval/yolov5n_head_int8
  source /home/developer/camera-auto-tracking/ArcFoundry.git/.venv/bin/activate
  python scripts/map_eval.py
  # or python scripts/map_eval.py 466  for full set, or python scripts/map_eval.py smoke for 10

WHY THIS MATTERS:
  - simulator cosine = 0.887 looks scary
  - but mAP might still be 0.80+ (only 3% drop from 0.83 onnx baseline)
  - mAP is the real-world metric; cosine is just a layer-level noise indicator
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
from rknn.api import RKNN

# ════════════════════════════════════════════════════════════════
# HARDCODED PATHS — change here if files move
# ════════════════════════════════════════════════════════════════
ONNX_PATH     = "/development/src/ai/camera-auto-tracking/ArcFoundry.git/yolov5n_head.onnx"
RKNN_FP16     = "/home/developer/camera-auto-tracking/ArcFoundry.git/output/rv1126b_yolov5n_head_fp16_640x640_20260707_release/yolov5n_head_fp16.rknn"
RKNN_INT8     = "/home/developer/camera-auto-tracking/ArcFoundry.git/output/rv1126b_yolov5n_head_int8_640x640_20260707_release/yolov5n_head_int8.rknn"
IMG_DIR       = "/home/developer/camera-auto-tracking/yolov5/yolov5/datasets/head/images/valid"
LABEL_DIR     = "/home/developer/camera-auto-tracking/yolov5/yolov5/datasets/head/labels/valid"
WORKSPACE     = Path("/home/developer/camera-auto-tracking/ArcFoundry.eval/yolov5n_head_int8")
TARGET        = "rv1126b"
IMG_SIZE      = 640
CONF_THRESHOLDS = [0.10, 0.25, 0.50]  # yolov5 default 0.25, plus lower/higher for sensitivity

# ════════════════════════════════════════════════════════════════

# letterbox: resize image to IMG_SIZE x IMG_SIZE keeping aspect ratio, grey padding
def letterbox(im, new_shape=IMG_SIZE, color=(114, 114, 114)):
    h0, w0 = im.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    pad_w = new_shape - nw
    pad_h = new_shape - nh
    pad = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    if r != 1:
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, pad[1], pad[3], pad[0], pad[2],
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, pad

# yolov5 3-head output → per-detection list [x1,y1,x2,y2,conf,cls] in feature coords
def non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45,
                        max_det=300, strides=(8, 16, 32)):
    """preds: list of 3 tensors, each [1, 18, H, W] (channels-first)."""
    decoded = []
    for i, p in enumerate(preds):
        p = np.ascontiguousarray(p)
        bs, no, gh, gw = p.shape
        # 18 channels = 3 anchors * (4+1+1) where per-anchor order is [x, y, w, h, obj, cls]
        # layout: [bs, anchors*6, H, W] → [bs, anchors, 6, H, W]
        p = p.reshape(bs, 3, 6, gh, gw)
        yv, xv = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grid = np.stack((xv, yv), axis=0).reshape(1, 1, 2, gh, gw).astype(np.float32)
        # yolov5 7.0 export: raw logits (sigmoid NOT baked in)
        # xy decode: 2*sigmoid(t) - 0.5 + grid_xy
        # wh decode: (2*sigmoid(t))^2 * stride
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))
        p[:, :, 0:2] = 2 * sigmoid(p[:, :, 0:2]) - 0.5 + grid
        p[:, :, 2:4] = (2 * sigmoid(p[:, :, 2:4])) ** 2 * strides[i]
        # now: [bs, anchors, 6, H, W] — flatten anchors and H*W
        p = p.reshape(bs, -1, 6)  # [bs, anchors*gh*gw, 6]
        p = p.reshape(bs, -1, 6)
        decoded.append(p)
    z = np.concatenate(decoded, axis=1)[0]  # [N_total, 6]
    # obj and cls are RAW LOGITS, not probabilities. Apply sigmoid to both.
    obj = 1.0 / (1.0 + np.exp(-z[:, 4:5].astype(np.float32)))
    cls = 1.0 / (1.0 + np.exp(-z[:, 5:6].astype(np.float32)))
    conf = obj * cls
    keep = conf[:, 0] > conf_thres
    z = np.concatenate([z[:, :4], conf, np.zeros((len(z), 1), dtype=np.float32)], axis=1)[keep]
    if len(z) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    order = z[:, 4].argsort()[::-1][:max_det]
    z = z[order]
    boxes_xyxy = np.concatenate([z[:, 0:2] - z[:, 2:4] / 2,
                                 z[:, 0:2] + z[:, 2:4] / 2], axis=1)
    return _nms(boxes_xyxy, z[:, 4], iou_thres, max_det)

def _nms(boxes, scores, iou_thres, max_det):
    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return np.column_stack([boxes[keep], scores[keep], np.zeros(len(keep), dtype=np.float32)])

# 11-point interpolated AP (Pascal VOC style)
def voc_ap(rec, prec):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = rec >= t
        p = prec[mask].max() if mask.any() else 0
        ap += p / 11
    return ap

def compute_map(predictions, gt, iou_thres=0.5, nc=1):
    """predictions: dict[img_id] -> [N,6], gt: dict[img_id] -> [(cls,x1,y1,x2,y2),...]"""
    aps = []
    for c in range(nc):
        dets, gts, npos = [], {}, 0
        for img_id, det in predictions.items():
            for d in det:
                if int(d[5]) == c:
                    dets.append((img_id, float(d[4]), d[:4]))
            for g in gt.get(img_id, []):
                if g[0] == c:
                    npos += 1
                    gts.setdefault(img_id, []).append(g[1:])
        if npos == 0:
            continue
        dets.sort(key=lambda x: -x[1])
        tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
        matched = {k: np.zeros(len(v), dtype=bool) for k, v in gts.items()}
        for i, (img_id, conf, box) in enumerate(dets):
            g = gts.get(img_id, [])
            if not g:
                fp[i] = 1
                continue
            g = np.array(g)
            iou = _box_iou(np.array(box)[None], g)[0]
            j = iou.argmax()
            if iou[j] >= iou_thres and not matched[img_id][j]:
                tp[i] = 1
                matched[img_id][j] = True
            else:
                fp[i] = 1
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        rec = tp / max(npos, 1)
        prec = tp / np.maximum(tp + fp, 1e-9)
        aps.append(voc_ap(rec, prec))
    return float(np.mean(aps)) if aps else 0.0

def _box_iou(a, b):
    a = a[:, None, :]; b = b[None, :, :]
    inter = np.clip(np.minimum(a[..., 2:], b[..., 2:]) -
                    np.maximum(a[..., :2], b[..., :2]), 0, None).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).prod(-1)
    return inter / (area_a + area_b - inter + 1e-9)

# ════════════════════════════════════════════════════════════════
def unletterbox(dets, pad, r, h0, w0):
    if len(dets) == 0:
        return dets
    dets = dets.copy()
    dets[:, [0, 2]] = (dets[:, [0, 2]] - pad[0]) / r
    dets[:, [1, 3]] = (dets[:, [1, 3]] - pad[1]) / r
    dets[:, :4] = dets[:, :4].clip(min=0)
    dets[:, [0, 2]] = dets[:, [0, 2]].clip(max=w0)
    dets[:, [1, 3]] = dets[:, [1, 3]].clip(max=h0)
    return dets

def main():
    # parse arg: number of images, default 466 (full set)
    n = sys.argv[1] if len(sys.argv) > 1 else "466"
    if n == "smoke":
        n = 10
    else:
        n = int(n)

    out_dir = WORKSPACE / f"results_{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.txt", "w", buffering=1)
    def p(msg):
        print(msg); log.write(str(msg) + "\n")

    p(f"Workspace: {WORKSPACE}")
    p(f"Mode: {n} images, conf thresholds {CONF_THRESHOLDS}, target={TARGET}")
    p("")

    # load ONNX
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    p(f"[onnx] in={in_name}  out={out_names}")

    # discover images
    import glob
    img_paths = sorted(glob.glob(str(Path(IMG_DIR) / "*.jpg")))[:n]
    p(f"[data] {len(img_paths)} images")

    # load GT
    p("[gt] loading ground truth labels...")
    gt = {}
    for ip in img_paths:
        img_id = Path(ip).stem
        lab_path = Path(LABEL_DIR) / f"{img_id}.txt"
        h0, w0 = cv2.imread(ip).shape[:2]
        g = []
        if lab_path.exists():
            for line in lab_path.read_text().strip().splitlines():
                p5 = line.split()
                cls = int(p5[0])
                cx, cy, bw, bh = map(float, p5[1:5])
                g.append((cls, (cx - bw/2)*w0, (cy - bh/2)*h0, (cx + bw/2)*w0, (cy + bh/2)*h0))
        gt[img_id] = g
    n_gt = sum(len(v) for v in gt.values())
    p(f"[gt] {n_gt} head boxes across {len(gt)} images")
    p("")

    # pre-letterbox
    p("[prep] letterboxing images...")
    imgs = {}
    for ip in img_paths:
        img_id = Path(ip).stem
        im = cv2.imread(ip)
        h0, w0 = im.shape[:2]
        lb, r, pad = letterbox(im)
        imgs[img_id] = (im, h0, w0, lb, r, pad)
    p("")

    def run_onnx():
        all_preds = []
        for i, ip in enumerate(img_paths):
            img_id = Path(ip).stem
            _, h0, w0, lb, r, pad = imgs[img_id]
            x = lb[:, :, ::-1].astype(np.float32) / 255.0
            x = x.transpose(2, 0, 1)[None]
            outs = sess.run(out_names, {in_name: x})
            for conf in CONF_THRESHOLDS:
                dets = non_max_suppression(outs, conf_thres=conf)
                dets = unletterbox(dets, pad, r, h0, w0)
                all_preds.append((img_id, conf, dets))
            if (i+1) % 50 == 0 or i == len(img_paths)-1:
                p(f"  onnx {i+1}/{len(img_paths)}")
        return all_preds

    def run_rknn(rknn_path):
        rknn = RKNN(verbose=False)
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                    target_platform=TARGET)
        rknn.load_rknn(rknn_path)
        rknn.init_runtime()  # target=None → simulator (no adb needed)
        all_preds = []
        for i, ip in enumerate(img_paths):
            img_id = Path(ip).stem
            _, h0, w0, lb, r, pad = imgs[img_id]
            x = lb.astype(np.uint8)
            outs = rknn.inference(inputs=[x])
            for conf in CONF_THRESHOLDS:
                dets = non_max_suppression(outs, conf_thres=conf)
                dets = unletterbox(dets, pad, r, h0, w0)
                all_preds.append((img_id, conf, dets))
            if (i+1) % 50 == 0 or i == len(img_paths)-1:
                p(f"  rknn {i+1}/{len(img_paths)}")
        rknn.release()
        return all_preds

    summary = {}

    def report(name, preds):
        for conf in CONF_THRESHOLDS:
            subset = {img_id: det for img_id, c, det in preds if c == conf}
            ap50 = compute_map(subset, gt, 0.5)
            ap5095 = compute_map(subset, gt, np.linspace(0.5, 0.95, 10).mean())
            summary[f"{name}_conf{conf}"] = {"mAP@0.5": ap50, "mAP@0.5:0.95": ap5095}
            p(f"  {name} conf={conf:.2f}  mAP@0.5={ap50:.4f}  mAP@0.5:0.95={ap5095:.4f}")

    p("=" * 60)
    p("RUNNING: ONNX (ground truth baseline)")
    p("=" * 60)
    t0 = time.time()
    onnx_preds = run_onnx()
    p(f"[onnx] {time.time()-t0:.1f}s total")
    report("onnx", onnx_preds)
    p("")

    p("=" * 60)
    p("RUNNING: RKNN FP16 (simulator)")
    p("=" * 60)
    t0 = time.time()
    fp16_preds = run_rknn(RKNN_FP16)
    p(f"[fp16] {time.time()-t0:.1f}s total")
    report("fp16", fp16_preds)
    p("")

    p("=" * 60)
    p("RUNNING: RKNN INT8 (the one we want to verify)")
    p("=" * 60)
    t0 = time.time()
    int8_preds = run_rknn(RKNN_INT8)
    p(f"[int8] {time.time()-t0:.1f}s total")
    report("int8", int8_preds)
    p("")

    p("=" * 60)
    p("FINAL SUMMARY (mAP@0.5 across conf thresholds)")
    p("=" * 60)
    p(f"{'config':<20s} {'conf':>6s} {'mAP@0.5':>10s} {'mAP@0.5:0.95':>15s} {'drop@0.5':>10s}")
    p("-" * 65)
    for conf in CONF_THRESHOLDS:
        on = summary[f"onnx_conf{conf}"]["mAP@0.5"]
        on5 = summary[f"onnx_conf{conf}"]["mAP@0.5:0.95"]
        fp = summary[f"fp16_conf{conf}"]["mAP@0.5"]
        fp5 = summary[f"fp16_conf{conf}"]["mAP@0.5:0.95"]
        i8 = summary[f"int8_conf{conf}"]["mAP@0.5"]
        i85 = summary[f"int8_conf{conf}"]["mAP@0.5:0.95"]
        p(f"{'onnx (baseline)':<20s} {conf:>6.2f} {on:>10.4f} {on5:>15.4f} {'—':>10s}")
        p(f"{'fp16 (RKNN)':<20s} {conf:>6.2f} {fp:>10.4f} {fp5:>15.4f} {(fp-on)*100:>+9.2f}%")
        p(f"{'int8 (RKNN)':<20s} {conf:>6.2f} {i8:>10.4f} {i85:>15.4f} {(i8-on)*100:>+9.2f}%")
        p("-" * 65)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    p(f"\n[done] results in {out_dir}/")
    log.close()

if __name__ == "__main__":
    main()
