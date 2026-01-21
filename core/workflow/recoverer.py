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
from core.utils import logger
from core.rknn_adapter import RKNNAdapter


class PrecisionRecoverer:

    def __init__(self, global_config, output_dir):
        self.cfg = global_config
        self.output_dir = output_dir

    # --------------------------------------------------------------------------
    # Level 3: 精度恢复工作流 (混合量化)
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
        cfg_file = f"{model_prefix}.quantization.cfg"
        model_file = f"{model_prefix}.model"
        data_file = f"{model_prefix}.data"

        analysis_dir = os.path.join(self.output_dir, "analysis", model_name)
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

        return

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
