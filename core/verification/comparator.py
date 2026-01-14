import numpy as np
import onnxruntime as ort
from rknn.api import RKNN
from core.utils import logger
from typing import List, Dict, Any

class ModelComparator:
    """
    负责对比 ONNX 原始模型与 RKNN 模拟器的推理精度。
    由于 RV1126B 等平台的 .rknn 文件无法在 PC 模拟运行，
    本模块会执行 "Shadow Build" (影子编译) 来启动模拟器。
    """

    def __init__(self, target_platform: str):
        self.target_platform = target_platform
        self.rknn = RKNN(verbose=False)

    def prepare_simulator(self, onnx_path: str, input_shapes: List[List[int]], build_config: Dict[str, Any]):
        """
        在内存中重新编译模型以启动模拟器 (Simulator Mode)
        """
        logger.info(f"[Verify] Initializing Simulator Environment for {self.target_platform}...")

        # 1. Config
        self.rknn.config(target_platform=self.target_platform)

        # 2. Load ONNX
        if self.rknn.load_onnx(model=onnx_path, input_size_list=input_shapes) != 0:
            raise RuntimeError("Simulator: Load ONNX failed!")

        # 3. Build (Shadow Build)
        # 注意：这里我们继承主流程的 build 配置（如量化参数），确保模拟的一致性
        # 但我们强制 do_quantization=False 来验证 FP16 基线，或者根据 build_config 决定
        # V1.1 阶段我们先验证 FP16 连通性
        ret = self.rknn.build(
            do_quantization=build_config.get('quantization', {}).get('enabled', False),
            dataset=build_config.get('quantization', {}).get('dataset', None)
        )
        if ret != 0:
            raise RuntimeError("Simulator: Build failed!")

        # 4. Init Runtime (target=None 触发模拟器模式)
        if self.rknn.init_runtime(target=None) != 0:
            raise RuntimeError("Simulator: Init Runtime failed! (Is rknn-toolkit2 installed correctly?)")

        logger.info("[Verify] Simulator Ready.")

    def compare_with_onnx(self, onnx_path: str, inputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        执行双轨推理并计算相似度。
        """
        # 1. ONNX Inference
        logger.info("[Verify] Running Baseline ONNX Inference...")
        sess = ort.InferenceSession(onnx_path)
        onnx_input_names = [i.name for i in sess.get_inputs()]
        onnx_feed = {name: inputs[name] for name in onnx_input_names if name in inputs}
        onnx_outputs = sess.run(None, onnx_feed)

        # 2. RKNN Inference
        # Simulator 接收列表输入，顺序需与 load_onnx 时一致
        rknn_feed_list = [inputs[name] for name in onnx_input_names]

        logger.info("[Verify] Running RKNN Simulator Inference...")
        rknn_outputs = self.rknn.inference(inputs=rknn_feed_list, data_format='nchw')

        # 3. Compute Metrics
        metrics = {}
        onnx_output_names = [o.name for o in sess.get_outputs()]

        # 防止输出数量不一致崩溃
        min_len = min(len(onnx_outputs), len(rknn_outputs))

        for idx in range(min_len):
            name = onnx_output_names[idx]
            out_onnx = onnx_outputs[idx].flatten()
            out_rknn = rknn_outputs[idx].flatten()

            # 处理 NaN/Inf
            if not np.isfinite(out_onnx).all() or not np.isfinite(out_rknn).all():
                logger.warning(f"[Verify] Output '{name}' contains NaN/Inf!")
                metrics[name] = 0.0
                continue

            # Cosine Similarity
            dot_product = np.dot(out_onnx, out_rknn)
            norm_a = np.linalg.norm(out_onnx)
            norm_b = np.linalg.norm(out_rknn)

            if norm_a == 0 or norm_b == 0:
                # 向量为0，视为完全一致(1.0)或完全丢失(0.0)，视业务而定，这里偏保守给 1.0 如果都为0
                cos_sim = 1.0 if norm_a == norm_b else 0.0
            else:
                cos_sim = dot_product / (norm_a * norm_b)

            metrics[name] = cos_sim
            logger.info(f"   📊 Metric: Output '{name}' Cosine Similarity = {cos_sim:.6f}")

        self.rknn.release()
        return metrics

    @staticmethod
    def validate_metric(metrics: Dict[str, float], threshold=0.98) -> bool:
        all_pass = True
        for name, score in metrics.items():
            if score < threshold:
                all_pass = False
        return all_pass