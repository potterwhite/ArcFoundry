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
from core.workflow.recoverer import PrecisionRecoverer
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
        self.recoverer = PrecisionRecoverer(self.cfg, self.json_output_dir)

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
                logger.debug(f"score={score}, entering precision recovery...")
                self.recoverer._recover_precision(json_target_platform, json_model_name, processed_onnx_path,
                                                  rknn_output_path, json_input_shapes, final_json_build,
                                                  custom_string)

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
