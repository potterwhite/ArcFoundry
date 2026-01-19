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

    # --------------------------------------------------------------------------
    # Assist Methods
    # --------------------------------------------------------------------------
    def _prepare_build_from_json(self, model_name, onnx_path):
        """
           Keep main loop clean by extracting config preparation logic
        """
        json_build_duplicate = copy.deepcopy(self.cfg.get('build', {}))
        json_build_duplicate['quantization']['dataset'] = None

        if json_build_duplicate.get('quantization', {}).get('enabled', False):
            if "encoder" in model_name.lower():
                # only encoder models use full quantization
                try:
                    # === [Optimization] Check if dataset list already exists ===
                    # 假设生成的文件名为 calibration_list.txt (这取决于 Calibrator 的实现，通常是固定的)
                    expected_ds_path = os.path.join(self.workspace, "calibration_list.txt")

                    if os.path.exists(expected_ds_path):
                        logger.info(f"⏩ [SKIP] Found existing calibration dataset: {expected_ds_path}")
                        ds_path = expected_ds_path
                    else:
                        # 只有不存在时才生成
                        calibrator = CalibrationGenerator(self.cfg)
                        ds_path = calibrator.generate(onnx_path, self.workspace)
                    # ===========================================================

                    if ds_path and os.path.exists(ds_path):
                        json_build_duplicate['quantization']['dataset'] = ds_path
                    else:
                        json_build_duplicate['quantization']['enabled'] = False
                except:
                    json_build_duplicate['quantization']['enabled'] = False
            else:
                # Other models (decoder, joiner) utilize fp16 only
                json_build_duplicate['quantization']['enabled'] = False
        return json_build_duplicate

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # --------------------------------------------------------------------------
    # Level 1: Main Entrance
    # --------------------------------------------------------------------------
    def run(self):
        # a. Preparation -- Extract info from yaml config
        json_project_name = self.cfg.get("project", {}).get("name")
        json_target_platform = self.cfg.get("target", {}).get("platform")
        json_models = self.cfg.get("models", [])

        # b. Preparation -- Initialize Helper Modules
        module_downloader = ModelDownloader()
        module_preprocessor = Preprocessor(self.cfg)

        # c. Preparation -- Success Counter Initialization
        success_count = 0

        # d. Preparation -- Echo Startup Info
        logger.info("==============================================================")
        logger.info(f"=== Starting ArcFoundry Pipeline: {json_project_name} on {json_target_platform} ===")

        # e. Main Loop -- Process Each Model
        for json_model in json_models:

            # 1. Preparation -- Extract Single Model Info
            json_model_name = json_model["name"]
            json_model_path = json_model["path"]  # YAML里指定的目标本地路径
            json_model_url = json_model.get("url", None)  # 既然是可选的，就用 get
            json_strategies = json_model.get("preprocess", {})
            rknn_out_path = os.path.join(self.output_dir, f"{json_model_name}.rknn")
            json_input_shapes = json_model.get('input_shapes', None)

            # 2. Preparation -- Echo helper info
            logger.info(f"\n>>> Processing Model: {json_model_name}")

            # 3. Preparation -- Verify and Download Model
            if not module_downloader.ensure_model(json_model_path, json_model_url):
                logger.error(f"Skipping {json_model_name} due to missing input file and download failed.")
                continue

            # 4. Preparation -- Define string of ONNX model path
            processed_onnx_name = f"{json_model_name}.processed.onnx"
            processed_onnx_path = os.path.join(self.workspace, processed_onnx_name)
            logger.debug(f"ONNX model path: {processed_onnx_path}")

            # 5. Processing -- Preprocessing Stage
            #    doing so many operations with the original model
            #    and return the processed onnx model path back
            logger.info(f"\n===== I. Preprocessing =====")
            processed_onnx_path, custom_string = module_preprocessor.process(
                json_model_path,
                processed_onnx_path,
                json_strategies,
            )

            if not processed_onnx_path:
                logger.error(f"Preprocessing failed for {json_model_name}")
                continue

            # --- Stage 2: RKNN Conversion ---
            logger.info(f"\n===== II. Calibration Dataset =====")
            final_json_build = self._prepare_build_from_json(json_model_name, processed_onnx_path)

            # 4. 执行标准转换与评估 (Level 2)
            logger.info(f"\n===== III. ONNX -> RKNN Conversion & Precision Verification =====")
            score = self._convert_and_evaluate(json_target_platform, json_model_name, processed_onnx_path,
                                               rknn_out_path, json_input_shapes, final_json_build,
                                               custom_string, json_model)

            # 5. 决策点：如果精度不够，进入恢复流程 (Level 3)
            # 只有开启了量化，且分数低，才触发
            logger.info(f"\n===== IV. Precision Recovery =====")
            is_quant = final_json_build.get('quantization', {}).get('enabled', False)
            if is_quant and score < 0.99:
                self._recover_precision(json_target_platform, json_model_name, processed_onnx_path,
                                        rknn_out_path, json_input_shapes, final_json_build, custom_string)

            logger.info(f"<<< Completed: {json_model_name} <<<\n")
            time.sleep(1)

        logger.info(f"\n=== Pipeline Completed: {success_count}/{len(json_models)} models successful ===")
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

    # --------------------------------------------------------------------------
    # Level 3: 精度恢复工作流 (混合量化)
    # --------------------------------------------------------------------------
    # def _recover_precision(self, target_plat, model_name, onnx_path, output_path, input_shapes,
    #                        base_build_config, custom_string):
    #     """
    #     独立的“救援”流程。包含：交互询问 -> 生成配置 -> 重新编译。
    #     此时之前的 adapter 已经释放，这里完全创建新的。
    #     """
    #     # Preparation -- 1. Paths
    #     analysis_dir = os.path.join(self.output_dir, "analysis", model_name)
    #     error_analysis_path = os.path.join(analysis_dir, "error_analysis.txt")
    #     quant_config_path = os.path.join(analysis_dir, "hybrid_quant_config.json")

    #     # Preparation -- 2. Echo welcome info
    #     logger.info(f"🚑 Entering Accuracy Recovery Workflow for {model_name}...")

    #     # Processing -- 3. User Selects whether to do Hybrid Quantization
    #     print(f"\n[INTERVENTION] Accuracy is below threshold. Analysis saved to: {analysis_dir}")
    #     choice = input(f"   >>> Enable Hybrid Quantization (FP16 mix)? [y/n]: ").strip().lower()
    #     if choice != 'y':
    #         return

    #     # Processing -- 4. User Selects Strategy(Auto / Manual)
    #     print("\n   [SELECT STRATEGY]")
    #     print("   (a) Auto-Tune: Automatically set layers < threshold to float16.")
    #     print("   (m) Manual: Generate template, you edit JSON manually.")
    #     mode = input("   >>> Select mode [a/m] (default: a): ").strip().lower()

    #     # Processing -- 5. Get Auto Threshold if needed
    #     auto_threshold = None
    #     if mode == 'm':
    #         # Manual Mode
    #         pass  # auto_threshold remains None
    #     else:
    #         # Auto Mode
    #         thresh_input = input("   >>> Enter min cosine score threshold (default 0.99): ").strip()
    #         try:
    #             auto_threshold = float(thresh_input) if thresh_input else 0.99
    #         except ValueError:
    #             logger.warning("Invalid number, using default 0.99")
    #             auto_threshold = 0.99

    #     # Processing -- 6. Preparing Hybrid Quantization Config
    #     # 为了生成配置，我们需要一个临时的 adapter 实例
    #     # 这是一个干净的实例，只为了 export_config，用完即扔
    #     if not os.path.exists(quant_config_path):
    #         if os.path.exists(error_analysis_path):
    #             temp_adapter = RKNNAdapter(target_plat, verbose=False)
    #             success = temp_adapter.generate_quant_config(error_analysis_path, quant_config_path,
    #                                                          auto_threshold)
    #             temp_adapter.release()

    #             if success:
    #                 print(f"   [CREATED] Config template: {quant_config_path}")
    #             else:
    #                 logger.error("   Failed to create template. Aborting.")
    #                 return

    #         else:
    #             logger.error("   Error analysis report missing. Cannot generate template.")
    #             return
    #     else:
    #         print(f"   [FOUND] {quant_config_path}")

    #     # Processing -- 5. Final Gate before real doing hybrid quantization
    #     if auto_threshold is None:
    #         print(f"\n   !!! ACTION: Please edit {quant_config_path} now.")
    #         print(f"   Change 'int8' to 'float16' for sensitive layers.")
    #         input("   >>> Press [ENTER] when you are ready to re-build...")
    #     else:
    #         print(f"   [AUTO] Applied settings for layers < {auto_threshold}. Re-building immediately...")

    #     # Processing -- 6. Determine final build config
    #     hybrid_build_config = copy.deepcopy(base_build_config)
    #     hybrid_build_config['quantization']['hybrid_config_path'] = quant_config_path

    #     # Processing -- 7. Do Hybrid Quantization Build
    #     final_adapter = RKNNAdapter(target_plat, verbose=True)
    #     ret = final_adapter.convert(onnx_path, output_path, input_shapes, hybrid_build_config, custom_string)

    #     # Processing -- 8. Prompt final result
    #     if ret:
    #         logger.info(f"✅ Hybrid Model successfully saved to {output_path}")
    #     else:
    #         logger.error(f"❌ Hybrid Conversion failed.")

    #     # Processing -- 9. Cleanup and exit
    #     final_adapter.release()
    def _recover_precision(self, target_plat, model_name, onnx_path, output_path, input_shapes,
                           base_build_config, custom_string):
        """
        Hybrid Quantization Workflow (The "Two-Step" Approach).
        """
        logger.info(f"🚑 Entering Accuracy Recovery Workflow for {model_name}...")

        # 0. Prepare Paths
        # RKNN generates files based on the ONNX filename in the current working directory
        # e.g., if onnx is "encoder.processed.onnx", it generates "encoder.processed.quantization.cfg"
        onnx_basename = os.path.basename(onnx_path)
        model_prefix = os.path.splitext(onnx_basename)[0]

        # Predicted paths for generated files (in CWD)
        cfg_file = f"{model_prefix}.quantization.cfg"
        model_file = f"{model_prefix}.model"
        data_file = f"{model_prefix}.data"

        analysis_dir = os.path.join(self.output_dir, "analysis", model_name)
        error_report = os.path.join(analysis_dir, "error_analysis.txt")

        # 1. Ask User
        print(f"\n[INTERVENTION] Accuracy is below threshold.")
        choice = input(f"   >>> Start Hybrid Quantization Step 1? [y/n]: ").strip().lower()
        if choice != 'y':
            return

        # 2. Step 1: Generate Intermediate Files
        # We need a fresh adapter
        adapter = RKNNAdapter(target_plat, verbose=True)
        adapter.config(base_build_config, custom_string)
        if not adapter.load_onnx(onnx_path, input_shapes):
            logger.error("Failed to load ONNX for hybrid step.")
            adapter.release()
            return

        dataset_path = base_build_config.get('quantization', {}).get('dataset')
        if not adapter.hybrid_step1(dataset_path):
            logger.error("Hybrid Step 1 failed.")
            adapter.release()
            return

        logger.info(f"   ✨ Step 1 Complete. Config generated at: ./{cfg_file}")

        # 3. Modify the Config (Auto vs Manual)
        print("\n   [SELECT STRATEGY]")
        print("   (a) Auto-Patch: Automatically set layers < threshold to float16.")
        print("   (m) Manual: You edit the .cfg file yourself.")
        mode = input("   >>> Select mode [a/m] (default: a): ").strip().lower()

        if mode == 'm':
            print(f"\n   !!! ACTION: Please edit ./{cfg_file} now.")
            print(f"   Find sensitive layers and change 'asymmetric_quantized-8' to 'float16'.")
            input("   >>> Press [ENTER] when you are ready for Step 2...")
        else:
            # Auto Mode
            thresh_input = input("   >>> Enter min cosine score threshold (default 0.99): ").strip()
            try:
                threshold = float(thresh_input) if thresh_input else 0.99
            except ValueError:
                threshold = 0.99

            # Call the patching method we just added to Adapter
            adapter.apply_hybrid_patch(cfg_file, error_report, threshold)

        # 4. Step 2: Build Final Model
        logger.info(f"🔄 Executing Hybrid Step 2...")
        if adapter.hybrid_step2(model_file, data_file, cfg_file):
            # 5. Export
            if adapter.export(output_path):
                 logger.info(f"✅ Hybrid Model successfully saved to {output_path}")
            else:
                 logger.error("❌ Export failed after hybrid build.")
        else:
            logger.error(f"❌ Hybrid Step 2 failed.")

        # Cleanup
        adapter.release()

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
