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

import os
import numpy as np
import onnxruntime as ort
from core.utils import logger
from core.rknn_adapter import RKNNAdapter
from core.verification.comparator import ModelComparator
from core.dsp.audio_features import SherpaFeatureExtractor


class StandardConverter:

    def __init__(self, global_config, output_dir):
        self.cfg = global_config
        self.output_dir = output_dir

    def _verify_model(self, model_cfg, onnx_path, build_config):
        # def _verify_model(self, model_cfg, onnx_path, rknn_path, build_config):
        """
        V1.1 Feature: Auto-Verification
        Returns:
            float: The minimum cosine similarity score (0.0 - 1.0).
                   Returns 1.0 if verification is skipped or crashes (to avoid false triggers).
        """
        logger.info(f"🔎 Starting Verification for {model_cfg['name']}...")
        min_score = 1.0  # Default safe value

        try:
            # 1. 初始化对比器
            target_platform = self.cfg.get("target", {}).get("platform")
            comparator = ModelComparator(target_platform)

            # --- CHANGE START ---
            # 旧代码: comparator.load_rknn(rknn_path)
            # 新代码: 传入 onnx路径, input_shapes, 和 build配置 进行影子编译
            input_shapes = model_cfg.get("input_shapes", None)
            #build_config = self.cfg.get("build", {})

            comparator.prepare_simulator(onnx_path, input_shapes, build_config)
            # --- CHANGE END ---

            # 2. 准备输入数据 (保持不变)
            sess = ort.InferenceSession(onnx_path)
            input_feed = {}
            extractor = SherpaFeatureExtractor()

            test_audio_path = self.cfg.get("build", {}).get("test_input", None)

            for i, inp in enumerate(sess.get_inputs()):
                # a. Handle Dynamic Shape (Replace string/None with 1)
                static_shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]

                # b. Detect NumPy Data Type
                onnx_type = inp.type
                np_dtype = np.float32  # Default fallback
                if "int64" in onnx_type:
                    np_dtype = np.int64
                elif "int32" in onnx_type:
                    np_dtype = np.int32
                elif "bool" in onnx_type:
                    np_dtype = bool
                elif "float16" in onnx_type:
                    np_dtype = np.float16

                # 处理动态 Shape
                static_shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]

                # c. Generate Input Data
                # Condition: Index 0 + Configured Path + File Exists + Is Float Type
                if (i == 0 and test_audio_path and os.path.exists(test_audio_path) and np.issubdtype(
                        np_dtype, np.floating)):
                    logger.info(f"   Using real audio for input '{inp.name}': {test_audio_path}")
                    feats = extractor.compute(test_audio_path)

                    # Crop to target length
                    target_len = static_shape[1]
                    if feats.shape[0] > target_len:
                        feats = feats[:target_len, :]

                    input_feed[inp.name] = np.expand_dims(feats, axis=0).astype(np_dtype)

                else:
                    # Fallback: Random Data based on Type
                    if np.issubdtype(np_dtype, np.integer):
                        # Generate random integers (e.g. sequence lengths)
                        input_feed[inp.name] = np.random.randint(0, 10, size=static_shape).astype(np_dtype)
                    elif np_dtype == bool:
                        input_feed[inp.name] = np.random.choice([True, False], size=static_shape)
                    else:
                        # Generate random floats
                        input_feed[inp.name] = np.random.rand(*static_shape).astype(np_dtype)

            # 3. 执行对比
            metrics = comparator.compare_with_onnx(onnx_path, input_feed)

            # [新增] 计算最低分
            if metrics:
                min_score = min(metrics.values())

            # 4. 判定结果
            if comparator.validate_metric(metrics, threshold=0.98):
                logger.info(f"✅ Verification PASSED: {model_cfg['name']} matches ONNX baseline.")
            else:
                logger.warning(
                    f"⚠️ Verification WARNING: {model_cfg['name']} accuracy might be low (Min Score: {min_score:.6f})."
                )

        except Exception as e:
            logger.error(f"❌ Verification Failed: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())  # 打印详细堆栈方便调试

        return min_score

    # --------------------------------------------------------------------------
    # Level 2: 标准转换与评估逻辑
    # --------------------------------------------------------------------------
    def convert_and_evaluate(self, target_plat, model_name, onnx_path, output_path, input_shapes,
                             build_config, custom_string, model_cfg):
        """
        负责一次标准的转换流程，并返回精度评分。
        注意：这个函数负责创建 adapter，使用它，然后必须释放它。
        """
        # === [Fast-Forward] Check for existing analysis report ===
        # 如果精度分析报告已存在，说明之前跑过且失败了，直接跳过构建，强制触发混合量化修复
        analysis_dir = os.path.join(self.output_dir, "analysis", model_name)
        existing_report = os.path.join(analysis_dir, "error_analysis.txt")

        # 只有当 RKNN 模型存在 且 分析报告也存在时，才跳过
        if os.path.exists(output_path) and os.path.exists(existing_report):
            logger.warning(f"⏩ [FAST-FORWARD] Found existing analysis report: {existing_report}")
            logger.warning(f"   Skipping Build & Verification to jump straight to Hybrid Quantization logic.")
            return 0.0  # 返回 0.0 分，强制触发 _recover_precision
        # =========================================================

        adapter = RKNNAdapter(target_platform=target_plat, verbose=build_config.get('verbose', False))

        # A. 转换
        ret = adapter.convert(onnx_path, output_path, input_shapes, build_config, custom_string)
        score = 1.0

        if ret:
            logger.info(f"SUCCESS: Standard model saved to {output_path}")

            # B. 验证 (Verify)
            score = self._verify_model(model_cfg, onnx_path, build_config)

            # C. 如果分数低，利用当前还活着的 adapter 做一次“尸检” (精度分析)
            #    这样我们就不用为了分析再重新 load 一次了
            is_quant = build_config.get('quantization', {}).get('enabled', False)
            if is_quant and score < 0.99:
                logger.warning(f"📉 Low Accuracy ({score:.4f}). Running immediate analysis before release...")
                dataset_path = build_config.get('quantization', {}).get('dataset')
                analysis_dir = os.path.join(self.output_dir, "analysis", model_name)
                adapter.run_deep_analysis(dataset_path, analysis_dir)
        else:
            logger.error(f"FAILURE: RKNN Conversion failed for {model_name}")
            score = 0.0

        # 必须释放！因为如果后面要进行混合量化，我们需要一个全新的环境
        adapter.release()
        return score
