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
from core.utils import logger, ensure_dir

class QuantizationAnalyzer:
    """
    v0.5.0 Feature: Quantization Accuracy Analyzer (The "CT Scanner")

    This module is responsible for diagnosing layer-wise accuracy degradation.
    It utilizes the `rknn.accuracy_analysis` API to compare FP32 vs INT8
    inference results layer by layer.
    """

    def __init__(self, rknn_instance, config):
        self.rknn = rknn_instance
        self.cfg = config

    def run(self, output_dir, dataset_path):
        """
        Execute the accuracy analysis.

        Args:
            output_dir (str): Directory to save the snapshot/report.
            dataset_path (str): Path to the 'dataset_list.txt' generated during calibration.
        """

        logger.info("################\n")
        logger.info("🩺 Starting Quantization Accuracy Analysis (This may take a while)...")

        # 2. Prepare Input Data
        # 'accuracy_analysis' requires a list of specific file paths, not the dataset txt file.
        # We grab the first sample (first line) from the dataset list.
        target_input = None
        if dataset_path and os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        # The dataset list format is "path1.npy path2.npy ..."
                        # Split it to get a list of inputs for multi-input models.
                        target_input = first_line.split(' ')
            except Exception as e:
                logger.error(f"   Failed to read dataset list: {e}")

        if not target_input:
            logger.warning("   Skipping analysis: Could not find valid input from dataset list.")
            return

        # 3. Setup Snapshot Directory
        snapshot_dir = os.path.join(output_dir, "accuracy_snapshot")
        ensure_dir(snapshot_dir)

        # 4. Invoke RKNN Analysis Tool
        # target=None implies running on the PC Simulator.
        try:
            ret = self.rknn.accuracy_analysis(
                inputs=target_input,
                output_dir=snapshot_dir,
                target=None,
                device_id=None
            )

            if ret == 0:
                logger.info(f"✅ Accuracy Analysis completed.")
                logger.info(f"   Snapshot saved to: {snapshot_dir}")
                logger.info("   Please check 'error_analysis.txt' in that directory for layer details.")
            else:
                logger.error("❌ Accuracy Analysis failed (internal rknn error).")

        except Exception as e:
            logger.error(f"❌ Accuracy Analysis crashed: {e}")
        logger.info("################\n")

