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

# core/quantization/configurator.py

import os
import copy
import json
import sys
from core.utils.utils import logger, timed_input
from core.quantization.calibrator import CalibrationGenerator


class QuantizationConfigurator:
    """
    Manages the quantization configuration logic and dependency resolution.
    Decides IF quantization should be enabled and PREPARES the necessary resources.
    """

    def __init__(self, global_config):
        """
        Args:
            global_config (dict): The full loaded yaml configuration.
            workspace_dir (str): Path to the workspace directory.
        """
        self.cfg = global_config
        self.workspace_dir = self.cfg.get("project", {}).get("workspace_dir", "./workspace")

    def _handle_fallback(self, reason):
        """
        Handles the missing dataset situation:
        1. Alerts the user.
        2. Waits 30s for decision (Exit or Continue as FP16).
        3. Exits program if user chooses 'n'.
        """
        # Preparation -- 1. Define local variables
        timeout = 30
        msg = [
            "\n" + "!" * 60, "!!! QUANTIZATION DATASET MISSING !!!", "!" * 60, f"Reason : {reason}",
            "!" * 60 + "\n"
        ]

        # Preparation -- 2. Print alert message
        logger.info("\n".join(msg))

        logger.info("⚠️  Action Required:")
        logger.info("   [Y] Downgrade to FP16 and continue (Default)")
        logger.info("   [N] Abort pipeline immediately")

        # Processing -- 1. Wait for user decision with timeout
        choice = timed_input(">>> Accept FP16 fallback? [Y/n]", timeout, default='y')

        # Processing -- 2. Act based on user choice
        if choice == 'n':
            logger.error("⛔ Pipeline aborted by user choice.")
            sys.exit(1)

        logger.warning("⚠️  Proceeding with FP16 Fallback mode...")

    def _get_dataset_path(self, onnx_path):
        # Preparation -- 2. Define expected calibration dataset path
        # The name of the calibration dataset file is fixed to "calibration_list.txt"
        expected_ds_path = os.path.join(self.workspace_dir, "calibration_list.txt")

        # === [Optimization] Check if dataset list already exists ===
        if os.path.exists(expected_ds_path):
            logger.info(f"⏩ [SKIP] Found existing calibration dataset: {expected_ds_path}")
            return_ds_path = expected_ds_path
        else:
            # generate if not exists
            calibrator = CalibrationGenerator(self.cfg)
            return_ds_path = calibrator.generate(onnx_path, self.workspace_dir)

        return return_ds_path

    def configure(self, model_name, onnx_path):
        """
        Keep main loop clean by extracting config preparation logic

        Args:
            model_name (str): Name of the model being processed.
            onnx_path (str): Path to the ONNX model file.

        Returns:
            dict: The final build configuration for this model.
        """
        logger.info(f"Configuring quantization settings for model: {model_name}")

        # Preparation -- 1. Duplicate build config for safety and modification
        json_build_duplicate = copy.deepcopy(self.cfg.get('build', {}))
        json_build_duplicate['quantization']['dataset'] = None

        # Preparation -- 2. Check if quantization is enabled
        if not json_build_duplicate.get('quantization', {}).get('enabled', False):
            return json_build_duplicate

        # Preparation -- 3. Check validity based on model name
        if "encoder" not in model_name.lower():
            # Other models (decoder, joiner) utilize fp16 only
            json_build_duplicate['quantization']['enabled'] = False
            return json_build_duplicate

        # Processing -- 2. Try to get dataset path
        try:
            ds_path = self._get_dataset_path(onnx_path)

            if ds_path and os.path.exists(ds_path):
                json_build_duplicate['quantization']['dataset'] = ds_path
                logger.info(
                    f"✅ Calibration dataset confirmed: {json_build_duplicate['quantization']['dataset']}")
            else:
                # Case A: Generator returned None or file doesn't exist
                reason = f"Dataset generator returned invalid path: {ds_path}"
                self._handle_fallback(reason)
                json_build_duplicate['quantization']['enabled'] = False

        except Exception as e:
            # Case B: Generator crashed (e.g., config error, missing file)
            # Print the stack trace for debugging, but don't crash the pipeline

            error_msg = f"Calibration Generator crashed: {str(e)}"
            # import traceback
            # logger.error(traceback.format_exc())

            self._handle_fallback(error_msg)
            json_build_duplicate['quantization']['enabled'] = False

        # Debug: Log final quantization config
        debug_msg = json.dumps(json_build_duplicate, indent=4, ensure_ascii=False)
        logger.warning(f"🔧 Final Quant Config:\n{debug_msg}")

        return json_build_duplicate
