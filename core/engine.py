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
import yaml
from core.utils import logger, ensure_dir
from core.preprocessor import Preprocessor
from core.rknn_adapter import RKNNAdapter
from core.downloader import ModelDownloader  # <--- 新增引用
import numpy as np
import onnxruntime as ort
from core.dsp.audio_features import SherpaFeatureExtractor
from core.verification.comparator import ModelComparator
from core.quantization.calibrator import CalibrationGenerator
import time
import copy


class PipelineEngine:
    """
    Orchestrates the conversion pipeline:
    Config -> Download(Optional) -> Preprocess -> Convert -> Output
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = self._load_config(config_path)

        # Paths
        self.workspace = self.cfg.get("project", {}).get("workspace_dir", "./workspace")
        self.output_dir = self.cfg.get("project", {}).get("output_dir", "./output")

        ensure_dir(self.workspace)
        ensure_dir(self.output_dir)

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def run(self):
        project_name = self.cfg.get("project", {}).get("name")
        target_plat = self.cfg.get("target", {}).get("platform")

        logger.info(f"=== Starting ArcFoundry Pipeline: {project_name} on {target_plat} ===")

        # Initialize Helper Modules
        downloader = ModelDownloader()
        preprocessor = Preprocessor(self.cfg)

        models = self.cfg.get("models", [])
        success_count = 0

        for model_cfg in models:
            model_name = model_cfg["name"]
            target_path = model_cfg["path"]  # YAML里指定的目标本地路径
            model_url = model_cfg.get("url", None)  # 既然是可选的，就用 get

            logger.info(f"\n>>> Processing Model: {model_name}")

            # --- Stage 0: Asset Management ---
            # Ensure model file is present (download if URL provided)
            if not downloader.ensure_model(target_path, model_url):
                logger.error(f"Skipping {model_name} due to missing input file.")
                continue

            # --- Stage 1: Preprocessing ---
            processed_onnx_name = f"{model_name}.processed.onnx"
            processed_onnx_path = os.path.join(self.workspace, processed_onnx_name)

            strategies = model_cfg.get("preprocess", {})

            final_onnx_path, custom_string = preprocessor.process(
                target_path,
                processed_onnx_path,
                strategies,
            )

            if not final_onnx_path:
                logger.error(f"Preprocessing failed for {model_name}")
                continue

            # --- Stage 2: RKNN Conversion ---
            rknn_out_path = os.path.join(self.output_dir, f"{model_name}.rknn")
            input_shapes = model_cfg.get('input_shapes', None)
            build_config = self._prepare_build_config(model_name, final_onnx_path)

            # 4. 执行标准转换与评估 (Level 2)
            score = self._convert_and_evaluate(target_plat, model_name, final_onnx_path, rknn_out_path,
                                               input_shapes, build_config, custom_string, model_cfg)

            # 5. 决策点：如果精度不够，进入恢复流程 (Level 3)
            # 只有开启了量化，且分数低，才触发
            is_quant = build_config.get('quantization', {}).get('enabled', False)
            if is_quant and score < 0.99:
                self._recover_precision(target_plat, model_name, final_onnx_path, rknn_out_path, input_shapes,
                                        build_config, custom_string)

            logger.info(f"<<< Completed: {model_name} <<<\n")
            time.sleep(1)

        logger.info(f"\n=== Pipeline Completed: {success_count}/{len(models)} models successful ===")
        logger.info("==============================================================")

    # --------------------------------------------------------------------------
    # Level 2: 标准转换与评估逻辑
    # --------------------------------------------------------------------------
    def _convert_and_evaluate(self, target_plat, model_name, onnx_path, output_path, input_shapes,
                              build_config, custom_string, model_cfg):
        """
        负责一次标准的转换流程，并返回精度评分。
        注意：这个函数负责创建 adapter，使用它，然后必须释放它。
        """
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

    # --------------------------------------------------------------------------
    # Level 3: 精度恢复工作流 (混合量化)
    # --------------------------------------------------------------------------
    def _recover_precision(self, target_plat, model_name, onnx_path, output_path, input_shapes,
                           base_build_config, custom_string):
        """
        独立的“救援”流程。包含：交互询问 -> 生成配置 -> 重新编译。
        此时之前的 adapter 已经释放，这里完全创建新的。
        """
        analysis_dir = os.path.join(self.output_dir, "analysis", model_name)
        error_analysis_path = os.path.join(analysis_dir, "error_analysis.txt")

        logger.info(f"\n🚑 Entering Accuracy Recovery Workflow for {model_name}...")

        # 1. 交互询问
        print(f"\n[INTERVENTION] Accuracy is below threshold. Analysis saved to: {analysis_dir}")
        choice = input(f"   >>> Enable Hybrid Quantization (FP16 mix)? [y/n]: ").strip().lower()
        if choice != 'y':
            return

        # 2. 准备混合量化配置
        quant_config_path = os.path.join(analysis_dir, "hybrid_quant_config.json")

        # 为了生成配置，我们需要一个临时的 adapter 实例
        # 这是一个干净的实例，只为了 export_config，用完即扔
        if not os.path.exists(quant_config_path):
            # logger.info("   Generating template config...")
            if os.path.exists(error_analysis_path):
                temp_adapter = RKNNAdapter(target_plat, verbose=False)
                success = temp_adapter.generate_quant_config(onnx_path, input_shapes, quant_config_path)
                temp_adapter.release()

                if success:
                    print(f"   [CREATED] Config template: {quant_config_path}")
                else:
                    logger.error("   Failed to create template. Aborting.")
                    return
            else:
                logger.error("   Error analysis report missing. Cannot generate template.")
                return
        else:
            print(f"   [FOUND] {quant_config_path}")


        # 3. 等待用户操作
        print(f"\n   !!! ACTION: Please edit {quant_config_path} now.")
        print(f"   Change 'int8' to 'float16' for sensitive layers (e.g. 7206-rs).")
        input("   >>> Press [ENTER] when you are ready to re-build...")

        # 4. 执行混合量化转换
        logger.info(f"🔄 Re-building with Hybrid Config...")

        # 注入配置路径
        hybrid_build_config = copy.deepcopy(base_build_config)
        hybrid_build_config['quantization']['hybrid_config_path'] = quant_config_path

        # 创建用于实际转换的新 adapter
        final_adapter = RKNNAdapter(target_plat, verbose=True)
        ret = final_adapter.convert(onnx_path, output_path, input_shapes, hybrid_build_config, custom_string)

        if ret:
            logger.info(f"✅ Hybrid Model successfully saved to {output_path}")
        else:
            logger.error(f"❌ Hybrid Conversion failed.")

        final_adapter.release()

    # --------------------------------------------------------------------------
    # Assist Methods
    # --------------------------------------------------------------------------
    def _prepare_build_config(self, model_name, onnx_path):
        """
           Keep main loop clean by extracting config preparation logic
        """
        build_config = copy.deepcopy(self.cfg.get('build', {}))
        build_config['quantization']['dataset'] = None

        if build_config.get('quantization', {}).get('enabled', False):
            if "encoder" in model_name.lower():
                # only encoder models use full quantization
                try:
                    calibrator = CalibrationGenerator(self.cfg)
                    ds_path = calibrator.generate(onnx_path, self.workspace)
                    if ds_path and os.path.exists(ds_path):
                        build_config['quantization']['dataset'] = ds_path
                    else:
                        build_config['quantization']['enabled'] = False
                except:
                    build_config['quantization']['enabled'] = False
            else:
                # Other models (decoder, joiner) utilize fp16 only
                build_config['quantization']['enabled'] = False
        return build_config

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
