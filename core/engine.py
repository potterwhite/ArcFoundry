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
from core.quantization.configurator import QuantizationConfigurator
from core.workflow.converter import StandardConverter
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
        self.json_workspace = self.cfg.get("project", {}).get("workspace_dir", "./workspace")
        self.json_output_dir = self.cfg.get("project", {}).get("output_dir", "./output")

        ensure_dir(self.json_workspace)
        ensure_dir(self.json_output_dir)

        self.quant_configurator = QuantizationConfigurator(self.cfg, self.json_workspace)
        self.converter = StandardConverter(self.cfg, self.json_output_dir)

    # --------------------------------------------------------------------------
    # Assist Methods
    # --------------------------------------------------------------------------
    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # --------------------------------------------------------------------------
    # Level 1: Main Entrance
    # --------------------------------------------------------------------------
    def run(self):
        # Preparation -- 1. Extract info from yaml config
        json_project_name = self.cfg.get("project", {}).get("name")
        json_target_platform = self.cfg.get("target", {}).get("platform")
        json_models = self.cfg.get("models", [])

        # Preparation -- 2. Initialize Helper Modules
        module_downloader = ModelDownloader()
        module_preprocessor = Preprocessor(self.cfg)

        # Preparation -- 3. Success Counter Initialization
        success_count = 0
        FAILED_MODELS = []

        # Preparation -- 4. Echo Startup Info
        logger.info("==============================================================")
        logger.info(f"=== Starting ArcFoundry Pipeline: {json_project_name} on {json_target_platform} ===")

        # Processing -- 1. Main Loop -- Process Each Model
        for json_model in json_models:

            # Preparation -- a. Extract Single Model Info
            json_model_name = json_model["name"]
            json_model_path = json_model["path"]
            json_model_url = json_model.get("url", None)
            json_strategies = json_model.get("preprocess", {})
            rknn_output_path = os.path.join(self.json_output_dir, f"{json_model_name}.rknn")
            json_input_shapes = json_model.get('input_shapes', None)

            # Preparation -- b. Echo helper info
            logger.info(f"\n>>> Processing Model: {json_model_name}")

            # Preparation -- c. Make sure model file exists (Download if needed)
            if not module_downloader.ensure_model(json_model_path, json_model_url):
                logger.error(f"Skipping {json_model_name} due to missing input file and download failed.")
                continue

            # Preparation -- d. Define string of ONNX model path
            processed_onnx_name = f"{json_model_name}.processed.onnx"
            processed_onnx_path = os.path.join(self.json_workspace, processed_onnx_name)
            logger.debug(f"ONNX model path: {processed_onnx_path}")

            # Processing -- a. Preprocessing Stage
            #    To do many operations with the original model and return the processed onnx model path back
            logger.info(f"\n===== I. Preprocessing =====")
            processed_onnx_path, custom_string = module_preprocessor.preprocess(
                json_model_path,
                processed_onnx_path,
                json_strategies,
            )

            if not processed_onnx_path:
                logger.error(f"Preprocessing failed for {json_model_name}")
                continue

            # Processing -- b. Build Configuration Preparation
            logger.info(f"\n===== II. Calibration Dataset =====")
            final_json_build = self.quant_configurator.configure(json_model_name, processed_onnx_path)

            # Processing -- c. Execute Standard Conversion & Evaluation (Level 2)
            logger.info(f"\n===== III. ONNX -> RKNN Conversion & Precision Verification =====")
            # score = self._convert_and_evaluate(json_target_platform, json_model_name, processed_onnx_path,
            #                                    rknn_output_path, json_input_shapes, final_json_build,
            #                                    custom_string, json_model)
            score = self.converter.convert_and_evaluate(
                json_target_platform, json_model_name, processed_onnx_path, rknn_output_path,
                json_input_shapes, final_json_build, custom_string, json_model)

            # Processing -- d. Decision Point: If Precision is Low, Enter Recovery Flow (Level 3)
            #               Only trigger if quantization is enabled and score is low
            logger.info(f"\n===== IV. Precision Recovery =====")
            is_quant = final_json_build.get('quantization', {}).get('enabled', False)
            if is_quant and score < 0.99:
                self._recover_precision(json_target_platform, json_model_name, processed_onnx_path,
                                        rknn_output_path, json_input_shapes, final_json_build, custom_string)

            # Processing -- e. Finalize Single Model
            if os.path.exists(rknn_output_path):
                success_count += 1
                logger.info(f"✅ Completed Model: {json_model_name} -> {rknn_output_path}")
            else:
                FAILED_MODELS.append(json_model_name)
                logger.error(
                    f"❌ Model conversion progress done, but rknn model {json_model_name} is checked not exists.")

        # Processing -- 2. Final Summary
        logger.info(f"\n=== Pipeline Completed: {success_count}/{len(json_models)} models successful ===")
        if FAILED_MODELS:
            logger.info(f"\nFailed Models: {FAILED_MODELS}\n")
        logger.info("==============================================================")

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
    #     analysis_dir = os.path.join(self.json_output_dir, "analysis", model_name)
    #     error_analysis_path = os.path.join(analysis_dir, "error_analysis.txt")
    #     quant_config_path = os.path.join(analysis_dir, "hybrid_quant_config.json")

    #     # Preparation -- 2. Echo welcome info
    #     logger.info(f"🚑 Entering Accuracy Recovery Workflow for {model_name}...")

    #     # Processing -- 3. User Selects whether to do Hybrid Quantization
    #     logger.info(f"\n[INTERVENTION] Accuracy is below threshold. Analysis saved to: {analysis_dir}")
    #     choice = input(f"   >>> Enable Hybrid Quantization (FP16 mix)? [y/n]: ").strip().lower()
    #     if choice != 'y':
    #         return

    #     # Processing -- 4. User Selects Strategy(Auto / Manual)
    #     logger.info("\n   [SELECT STRATEGY]")
    #     logger.info("   (a) Auto-Tune: Automatically set layers < threshold to float16.")
    #     logger.info("   (m) Manual: Generate template, you edit JSON manually.")
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
    #                 logger.info(f"   [CREATED] Config template: {quant_config_path}")
    #             else:
    #                 logger.error("   Failed to create template. Aborting.")
    #                 return

    #         else:
    #             logger.error("   Error analysis report missing. Cannot generate template.")
    #             return
    #     else:
    #         logger.info(f"   [FOUND] {quant_config_path}")

    #     # Processing -- 5. Final Gate before real doing hybrid quantization
    #     if auto_threshold is None:
    #         logger.info(f"\n   !!! ACTION: Please edit {quant_config_path} now.")
    #         logger.info(f"   Change 'int8' to 'float16' for sensitive layers.")
    #         input("   >>> Press [ENTER] when you are ready to re-build...")
    #     else:
    #         logger.info(f"   [AUTO] Applied settings for layers < {auto_threshold}. Re-building immediately...")

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

        analysis_dir = os.path.join(self.json_output_dir, "analysis", model_name)
        error_report = os.path.join(analysis_dir, "error_analysis.txt")

        # 1. Ask User
        logger.info(f"\n[INTERVENTION] Accuracy is below threshold.")
        choice = input(f"   >>> Start Hybrid Quantization Step 1/2? [Y/n]: ").strip().lower()
        if choice not in ('', 'y', 'yes'):
            return
        else:
            logger.info(f"\n\n   🔄 Starting Hybrid Quantization Step 1/2...")

        # 2. Step 1/2: Generate Intermediate Files
        # We need a fresh adapter
        adapter = RKNNAdapter(target_plat, verbose=True)
        adapter.config(base_build_config, custom_string)
        if not adapter.load_onnx(onnx_path, input_shapes):
            logger.error("Failed to load ONNX for hybrid step.")
            adapter.release()
            return

        dataset_path = base_build_config.get('quantization', {}).get('dataset')
        if not adapter.hybrid_step1(dataset_path):
            logger.error("Hybrid Step 1/2 failed.")
            adapter.release()
            return

        logger.info(f"   ✨ Step 1/2 Complete. Config generated at: ./{cfg_file}")

        # 3. Modify the Config (Auto vs Manual)
        logger.info("\n   [SELECT STRATEGY]")
        logger.info("   (a) Auto-Patch: Automatically set layers < threshold to float16.")
        logger.info("   (m) Manual: You edit the .cfg file yourself.")
        mode = input("   >>> Select mode [a/m] (default: a): ").strip().lower()

        if mode == 'm':
            logger.info(f"\n   !!! ACTION: Please edit ./{cfg_file} now.")
            logger.info(f"   Find sensitive layers and change 'asymmetric_quantized-8' to 'float16'.")
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
