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
from core.utils import logger, timed_input
from core.rknn_adapter import RKNNAdapter


class PrecisionRecoverer:

    def __init__(self, global_config):
        self.cfg = global_config
        self.output_dir = self.cfg.get("project", {}).get("output_dir", "./output")

    # --------------------------------------------------------------------------
    # Main Methods -- Precision Recovery Workflow
    # --------------------------------------------------------------------------
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
        # # === [MODIFIED] Force paths to be inside the temp directory ===
        # cfg_file = os.path.join(self.output_dir, f"{model_prefix}.quantization.cfg")
        # model_file = os.path.join(self.output_dir, f"{model_prefix}.model")
        # data_file = os.path.join(self.output_dir, f"{model_prefix}.data")
        # ==============================================================
        cfg_file = f"{model_prefix}.quantization.cfg"
        model_file = f"{model_prefix}.model"
        data_file = f"{model_prefix}.data"

        analysis_dir = os.path.join(self.output_dir, "analysis", model_name)
        error_report = os.path.join(analysis_dir, "error_analysis.txt")

        # 1. Ask User
        logger.info(f"\n[INTERVENTION] Accuracy is below threshold.")
        # choice = input(f"   >>> Start Hybrid Quantization Step 1/2? [Y/n]: ").strip().lower()
        choice = timed_input("   >>> Start Hybrid Quantization Step 1/2? [Y/n]: ", timeout=15, default='y')
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

        # 3. Step 2/2: User Selection of Sensitive Layers
        logger.info("\n\n")
        logger.info("   ✨ Step 2/2 Started:")
        logger.info("   [SELECT STRATEGY]")
        logger.info("   (a) Auto-Patch: Automatically set layers < threshold to float16.")
        logger.info("   (m) Manual: You edit the .cfg file yourself.")
        # mode = input("   >>> Select mode [a/m] (default: a): ").strip().lower()
        mode = timed_input("   >>> Select mode [a/m] (default: a): ", timeout=10, default='a')
        # logger.error(f"\n\nmode: {mode}++++++\n\n")

        if mode == 'm':
            logger.info(f"\n   !!! ACTION: Please edit ./{cfg_file} now.")
            logger.info(f"   Find sensitive layers and change 'asymmetric_quantized-8' to 'float16'.")
            input("   >>> Press [ENTER] when you are ready for Step 2...")
        else:
            # Auto Mode
            # thresh_input = input("   >>> Enter min cosine score threshold (default 0.99): ").strip()
            thresh_input = timed_input("   >>> Enter min cosine score threshold (default 0.99): ",
                                       timeout=15,
                                       default='0.99')
            # logger.error(f"\n\nthresh_input: {thresh_input}++++++\n\n")

            try:
                threshold = float(thresh_input) if thresh_input else 0.99
            except ValueError:
                threshold = 0.99

            # Call the patching method we just added to Adapter
            adapter.apply_hybrid_patch(cfg_file, error_report, threshold)

        # 4. Step 2/2: Build Final Model
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

        return
