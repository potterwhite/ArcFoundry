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
from core.utils import logger

# Import the registry and strategies to ensure they are registered
from core.quantization.strategies import get_strategy_class
# NOTE: Importing the module below triggers the @register_strategy decorator
import core.quantization.strategies.streaming
# Future: import core.quantization.strategies.vision


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
                                        If None, defaults to 'streaming_audio' (Backward Compatibility).
        """
        self.cfg = config

        # Determine strategy: passed arg > config > default
        # You can add a 'type' field in your YAML under 'models' later.
        if not model_type:
            model_type = 'streaming_audio'

        self.model_type = model_type

        # Factory Pattern: Get the worker class
        try:
            strategy_cls = get_strategy_class(self.model_type)
            self.strategy = strategy_cls(config)
            logger.debug(f"Initialized Calibration Strategy: {self.model_type}")
        except ValueError as e:
            logger.error(str(e))
            raise

    def generate(self, onnx_path, output_dir):
        """
        Standard interface called by PipelineEngine.
        """
        # 1. Common Logic: Extract dataset path from config
        dataset_path = self.cfg.get('build', {}).get('quantization', {}).get('dataset')

        if not dataset_path:
            logger.warning("Quantization enabled but 'dataset' path missing in config.")
            return None

        # 2. Common Logic: Check Cache (Avoid re-running if exists)
        list_file_path = os.path.join(output_dir, "dataset_list.txt")
        if self._check_cache(list_file_path):
            return list_file_path

        # 3. Delegate to Strategy
        return self.strategy.run(onnx_path, output_dir, dataset_path)

    def _check_cache(self, list_file_path):
        """Simple smoke test to see if calibration data exists."""
        if os.path.exists(list_file_path):
            with open(list_file_path, 'r') as f:
                first_line = f.readline().strip()
                # Check if the file referenced in the first line actually exists
                if first_line and os.path.exists(first_line.split(' ')[0]):
                    logger.info(f"⏩ [FAST-FORWARD] Found existing calibration list: {list_file_path}")
                    return True
        return False
