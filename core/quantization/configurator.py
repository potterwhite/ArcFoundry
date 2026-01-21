# core/quantization/configurator.py

import os
import copy
import json
from core.utils import logger
from core.quantization.calibrator import CalibrationGenerator


class QuantizationConfigurator:
    """
    Manages the quantization configuration logic and dependency resolution.
    Decides IF quantization should be enabled and PREPARES the necessary resources.
    """

    def __init__(self, global_config, workspace_dir):
        """
        Args:
            global_config (dict): The full loaded yaml configuration.
            workspace_dir (str): Path to the workspace directory.
        """
        self.cfg = global_config
        self.workspace_dir = workspace_dir

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

            # 打印一下拿到的路径是什么，方便调试
            logger.debug(f"Calibration Generator returned: {ds_path}")

            if ds_path and os.path.exists(ds_path):
                json_build_duplicate['quantization']['dataset'] = ds_path
            else:
                logger.warning(f"⚠️ Dataset path is invalid: {ds_path}. Disabling quantization.")
                json_build_duplicate['quantization']['enabled'] = False

        except Exception as e:
            logger.error(f"❌ Dataset Generation Crashing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())  # 打印详细报错堆栈

            json_build_duplicate['quantization']['enabled'] = False

        # Debug: Log final quantization config
        debug_msg = json.dumps(json_build_duplicate, indent=4, ensure_ascii=False)
        logger.warning(f"🔧 Final Quant Config:\n{debug_msg}")

        return json_build_duplicate
