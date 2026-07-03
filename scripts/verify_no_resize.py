#!/usr/bin/env python3
# Copyright (c) 2026 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
验证 no-resize ONNX 与原始 ONNX 的输出一致性。
原始 ONNX 输出：fgr(1,3,H,W) + pha(1,1,H,W) + r1o~r4o
no-resize ONNX 输出：A(1,4,272,480) + b(1,4,272,480) + r1o~r4o
C++ guided filter 组合在 Python 中预演。

依赖：pip install onnxruntime numpy opencv-python
"""

import argparse
import logging

import cv2
import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_original(session, src, r1i, r2i, r3i, r4i, dsr):
    """Run original ONNX model (outputs fgr, pha, r1o~r4o)."""
    outputs = session.run(None, {
        "src": src, "r1i": r1i, "r2i": r2i, "r3i": r3i, "r4i": r4i,
        "downsample_ratio": np.array([dsr], dtype=np.float32),
    })
    # outputs: [fgr, pha, r1o, r2o, r3o, r4o]
    return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]


def run_no_resize(session, src, r1i, r2i, r3i, r4i, dsr):
    """Run no-resize ONNX model (outputs A, b, r1o~r4o)."""
    outputs = session.run(None, {
        "src": src, "r1i": r1i, "r2i": r2i, "r3i": r3i, "r4i": r4i,
        "downsample_ratio": np.array([dsr], dtype=np.float32),
    })
    # outputs order depends on graph.outputs order: [r1o, r2o, r3o, r4o, A, b]
    # Need to identify which is A and which is b by shape
    result = {}
    for i, out in enumerate(outputs):
        shape = out.shape
        if len(shape) == 4 and shape[1] == 4 and shape[2] == 272:
            # This is A or b (4ch, 272x480)
            if "A" not in result:
                result["A"] = out
            else:
                result["b"] = out
        elif len(shape) == 4 and shape[1] == 16:
            result["r1o"] = out
        elif len(shape) == 4 and shape[1] == 20:
            result["r2o"] = out
        elif len(shape) == 4 and shape[1] == 40:
            result["r3o"] = out
        elif len(shape) == 4 and shape[1] == 64:
            result["r4o"] = out

    return result["A"], result["b"], result["r1o"], result["r2o"], result["r3o"], result["r4o"]


def guided_filter_combine(A, b, src, dsr):
    """
    C++ guided filter combination (Python pre-implementation).
    A: (1,4,272,480), b: (1,4,272,480), src: (1,3,544,960) normalized [0,1]

    ONNX graph formula:
    1. fine_x = [src, src_mean] (4ch)
    2. out = A_up * fine_x_up + b_up (4ch)
    3. fgr = clip(out[:,:3] + src, 0, 1)  // Add_350
    4. pha = clip(out[:,3:], 0, 1)
    """
    H, W = src.shape[2], src.shape[3]

    # Upsample A and b to full resolution
    A_up = cv2.resize(A[0].transpose(1, 2, 0), (W, H), interpolation=cv2.INTER_LINEAR)
    A_up = A_up.transpose(2, 0, 1)[np.newaxis]  # (1,4,H,W)

    b_up = cv2.resize(b[0].transpose(1, 2, 0), (W, H), interpolation=cv2.INTER_LINEAR)
    b_up = b_up.transpose(2, 0, 1)[np.newaxis]  # (1,4,H,W)

    # fine_x = [src, src_mean] (4ch)
    src_mean = src.mean(axis=1, keepdims=True)  # (1,1,H,W)
    fine_x = np.concatenate([src, src_mean], axis=1)  # (1,4,H,W)

    # out = A_up * fine_x + b_up
    out = A_up * fine_x + b_up  # (1,4,H,W)

    # Split
    fgr_part = out[:, :3]  # (1,3,H,W)
    pha = out[:, 3:]  # (1,1,H,W)

    # Add_350: fgr = clip(fgr_part + src, 0, 1)
    fgr = np.clip(fgr_part + src, 0, 1)
    pha = np.clip(pha, 0, 1)

    return fgr, pha


def cosine_similarity(a, b):
    """Compute cosine similarity between two arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Path to original ONNX")
    parser.add_argument("--no-resize", required=True, help="Path to no-resize ONNX")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--dsr", type=float, default=0.5, help="downsample_ratio")
    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"Cannot read image: {args.image}")
        return

    H, W = 544, 960
    img = cv2.resize(img, (W, H))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    src = img_rgb.transpose(2, 0, 1)[np.newaxis]  # (1,3,H,W)

    # Init recurrent states
    dsr = args.dsr
    dH, dW = int(H * dsr), int(W * dsr)
    r1i = np.zeros((1, 16, dH // 2, dW // 2), dtype=np.float32)
    r2i = np.zeros((1, 20, dH // 4, dW // 4), dtype=np.float32)
    r3i = np.zeros((1, 40, dH // 8, dW // 8), dtype=np.float32)
    r4i = np.zeros((1, 64, dH // 16, dW // 16), dtype=np.float32)

    # Load sessions
    logger.info(f"Loading original: {args.original}")
    sess_orig = ort.InferenceSession(args.original)
    logger.info(f"Loading no-resize: {args.no_resize}")
    sess_nr = ort.InferenceSession(args.no_resize)

    # Run original
    logger.info("Running original ONNX...")
    fgr_orig, pha_orig, r1o_o, r2o_o, r3o_o, r4o_o = run_original(sess_orig, src, r1i, r2i, r3i, r4i, dsr)
    logger.info(f"  fgr: {fgr_orig.shape}, pha: {pha_orig.shape}")

    # Run no-resize
    logger.info("Running no-resize ONNX...")
    A, b, r1o_n, r2o_n, r3o_n, r4o_n = run_no_resize(sess_nr, src, r1i, r2i, r3i, r4i, dsr)
    logger.info(f"  A: {A.shape}, b: {b.shape}")

    # Guided filter combine
    logger.info("Applying guided filter combination...")
    fgr_nr, pha_nr = guided_filter_combine(A, b, src, dsr)
    logger.info(f"  fgr: {fgr_nr.shape}, pha: {pha_nr.shape}")

    # Compare
    pha_cos = cosine_similarity(pha_orig, pha_nr)
    fgr_cos = cosine_similarity(fgr_orig, fgr_nr)
    r1o_cos = cosine_similarity(r1o_o, r1o_n)
    r2o_cos = cosine_similarity(r2o_o, r2o_n)
    r3o_cos = cosine_similarity(r3o_o, r3o_n)
    r4o_cos = cosine_similarity(r4o_o, r4o_n)

    logger.info("=== Cosine Similarity ===")
    logger.info(f"  pha:  {pha_cos:.6f}")
    logger.info(f"  fgr:  {fgr_cos:.6f}")
    logger.info(f"  r1o:  {r1o_cos:.6f}")
    logger.info(f"  r2o:  {r2o_cos:.6f}")
    logger.info(f"  r3o:  {r3o_cos:.6f}")
    logger.info(f"  r4o:  {r4o_cos:.6f}")

    # Max absolute difference
    logger.info("=== Max Abs Diff ===")
    logger.info(f"  pha:  {np.max(np.abs(pha_orig - pha_nr)):.6f}")
    logger.info(f"  fgr:  {np.max(np.abs(fgr_orig - fgr_nr)):.6f}")

    # Threshold check
    ok = True
    if pha_cos < 0.99:
        logger.warning(f"pha cosine similarity {pha_cos:.6f} < 0.99")
        ok = False
    if fgr_cos < 0.95:
        logger.warning(f"fgr cosine similarity {fgr_cos:.6f} < 0.95")
        ok = False

    if ok:
        logger.info("PASS: Quality verification passed")
    else:
        logger.warning("FAIL: Quality verification failed")


if __name__ == "__main__":
    main()
