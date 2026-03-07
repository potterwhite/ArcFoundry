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

# core/quantization/calibrator.py

import os
from utils import logger

# Import the registry and strategies to ensure they are registered
from .strategies import get_strategy_class
# NOTE: Importing the module below triggers the @register_strategy decorator
import core.pipeline.ii_configuration.strategies.vision
import core.pipeline.ii_configuration.strategies.streaming


class CalibrationGenerator:
    """
    Facade class for generating calibration datasets.

    It delegates the actual data processing to a specific strategy
    (e.g., streaming_audio, static_image) based on the configuration.
    """

    def __init__(self, config, model_type=None):
        """
        Args:
            config (dict): Global configuration.
            model_type (str, optional): Explicit strategy name.
                                        If None, auto-detects based on model name.
        """
        self.cfg = config

        # Determine strategy: passed arg > auto-detection > default
        if not model_type:
            model_type = self._auto_detect_model_type()

        self.model_type = model_type

        # Factory Pattern: Get the worker class
        try:
            strategy_cls = get_strategy_class(self.model_type)
            self.strategy = strategy_cls(config)
            logger.info(
                f"✅ Initialized Calibration Strategy: {self.model_type}")
        except ValueError as e:
            logger.error(str(e))
            raise

    def _auto_detect_model_type(self):
        """
        Auto-detect model type based on model name in config.

        Returns:
            str: Strategy name ('vision' or 'streaming_audio')
        """
        models = self.cfg.get('models', [])
        if not models:
            logger.warning(
                "No models found in config, defaulting to 'streaming_audio'")
            return 'streaming_audio'

        model_name = models[0].get('name', '').lower()

        # CV model keywords
        cv_keywords = [
            'modnet', 'yolo', 'resnet', 'mobilenet', 'efficientnet',
            'segmentation', 'detection', 'classification', 'matting'
        ]

        # ASR model keywords
        asr_keywords = [
            'encoder', 'decoder', 'joiner', 'sherpa', 'zipformer',
            'transducer', 'conformer', 'asr'
        ]

        # Check for CV models
        if any(keyword in model_name for keyword in cv_keywords):
            logger.info(
                f"Auto-detected CV model: {model_name} -> using 'vision' strategy"
            )
            return 'vision'

        # Check for ASR models
        if any(keyword in model_name for keyword in asr_keywords):
            logger.info(
                f"Auto-detected ASR model: {model_name} -> using 'streaming_audio' strategy"
            )
            return 'streaming_audio'

        # Default to vision for unknown models
        logger.warning(
            f"Unknown model type: {model_name}, defaulting to 'vision' strategy"
        )
        return 'vision'

    def _check_cache(self, list_file_path):
        """Simple smoke test to see if calibration data exists."""
        if os.path.exists(list_file_path):
            with open(list_file_path, 'r') as f:
                first_line = f.readline().strip()
                # Check if the file referenced in the first line actually exists
                if first_line and os.path.exists(first_line.split(' ')[0]):
                    logger.info(
                        f"⏩ [FAST-FORWARD] Found existing calibration list: {list_file_path}"
                    )
                    return True
        return False

    def generate(self, onnx_path, output_dir):
        """
        Standard interface called by PipelineEngine.
        """
        # Preparation -- 1. Common Logic: Extract dataset path from config
        json_ds_path = self.cfg.get('build', {}).get('quantization',
                                                     {}).get('dataset')
        list_file_path = os.path.join(output_dir, "dataset_list.txt")

        # Preparation -- 2. Validate presence of dataset path
        if not json_ds_path:
            logger.warning(
                "Quantization enabled but 'dataset' path missing in config.")
            return None

        # Preparation -- 3. Common Logic: Check Cache (Avoid re-running if exists)
        if self._check_cache(list_file_path):
            return list_file_path

        # Main Processing -- Delegate to strategy
        #   Note : this run() actually in "core/quantization/strategies/xxxx.py" which is according to self.model_type we specified earlier
        return self.strategy.run(onnx_path, output_dir, json_ds_path)
