#!/usr/bin/env python3
"""
RVM Refiner Resize 移除脚本（独立，不依赖 ArcFoundry）
只移除 refiner 尾巴（Resize×2 + element-wise），保留 refiner Conv 网络在 NPU 上计算 A 和 b。

输入：rvm_mobilenetv3_fp32.onnx（原始模型）
输出：rvm_mobilenetv3_fp32_no_resize.onnx（A(4ch,272×480) + b(4ch,272×480) + r1o~r4o）

依赖：pip install onnx onnx-graphsurgeon
用法：
  python scripts/remove_rvm_refiner.py \
    --input models/downloads/rvm_mobilenetv3.onnx \
    --output models/downloads/rvm_mobilenetv3_no_resize.onnx
"""

import argparse
import logging
import sys

import numpy as np
import onnx
import onnx_graphsurgeon as gs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def remove_refiner(input_path: str, output_path: str):
    model = onnx.load(input_path)
    graph = gs.import_onnx(model)

    # 记录原始状态
    orig_node_count = len(graph.nodes)
    orig_output_names = [o.name for o in graph.outputs]
    logger.info(f"Original model: {orig_node_count} nodes, outputs={orig_output_names}")

    # 找到 refiner Conv 网络的输出 tensors
    # A: Conv_316 (refiner.conv.6) → output '777', shape (1,4,272,480)
    # b: Sub_318 (mean_y - A×mean_x) → output '779', shape (1,4,272,480)
    A_tensor = None
    b_tensor = None
    for node in graph.nodes:
        if node.name == "Conv_316":
            A_tensor = node.outputs[0]
        elif node.name == "Sub_318":
            b_tensor = node.outputs[0]

    if A_tensor is None:
        logger.error("Conv_316 (refiner A output) not found in graph")
        sys.exit(1)
    if b_tensor is None:
        logger.error("Sub_318 (refiner b output) not found in graph")
        sys.exit(1)

    # 设置 dtype（cleanup 后中间 tensor 可能丢失类型信息）
    A_tensor.dtype = np.float32
    b_tensor.dtype = np.float32
    logger.info(f"Found A tensor: '{A_tensor.name}' shape={A_tensor.shape}")
    logger.info(f"Found b tensor: '{b_tensor.name}' shape={b_tensor.shape}")

    # 替换 graph.outputs：移除 fgr/pha，添加 A 和 b
    new_outputs = []
    removed = []
    for out in graph.outputs:
        if out.name in ("fgr", "pha"):
            removed.append(out.name)
            continue
        new_outputs.append(out)
    new_outputs.append(A_tensor)
    new_outputs.append(b_tensor)

    logger.info(f"Removing outputs: {removed}")
    logger.info(f"Adding outputs: '{A_tensor.name}', '{b_tensor.name}'")

    graph.outputs = new_outputs
    graph.cleanup().toposort()

    # cleanup 后重新设置 dtype（可能被清除）
    for out in graph.outputs:
        if out.name in ("777", "779"):
            out.dtype = np.float32

    # 验证：确认 r1o~r4o 仍在输出中
    final_output_names = [o.name for o in graph.outputs]
    logger.info(f"After surgery: {len(graph.nodes)} nodes (was {orig_node_count})")
    logger.info(f"Final outputs: {final_output_names}")

    expected_remaining = [n for n in orig_output_names if n not in ("fgr", "pha")]
    missing = [n for n in expected_remaining if n not in final_output_names]
    if missing:
        logger.error(f"Missing expected outputs after cleanup: {missing}")
        logger.error("graph.cleanup() may have removed recurrent state paths!")
        sys.exit(1)

    # 保存
    model = gs.export_onnx(graph)
    onnx.save(model, output_path)
    logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove RVM DeepGuidedFilterRefiner from ONNX graph")
    parser.add_argument("--input", required=True, help="Path to original rvm_mobilenetv3.onnx")
    parser.add_argument("--output", required=True, help="Path for output (no-refiner) ONNX")
    args = parser.parse_args()
    remove_refiner(args.input, args.output)
