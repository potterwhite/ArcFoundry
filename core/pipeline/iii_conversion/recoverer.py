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
import re
import traceback
from utils import logger, timed_input
from ._rknn_adapter import RKNNAdapter


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

        # 4. Step 2/2: Build Final Model (with dtype retry logic)
        #
        # RKNN SDK may raise "dtype not allowed to be modified" for certain
        # tensors during hybrid_quantization_step2. The fix is NOT to change
        # the dtype value — it is to REMOVE the tensor's entry block entirely
        # from the .quantization.cfg, so RKNN falls back to its default from
        # the .model file. Each retry handles one locked tensor; we stop early
        # if the same tensor appears twice (would mean removal failed silently).
        MAX_DTYPE_RETRIES = 10
        step2_success = False
        removed_tensors = set()  # Guard: detect if same tensor appears twice

        for attempt in range(1, MAX_DTYPE_RETRIES + 1):
            logger.info(f"🔄 Executing Hybrid Step 2 (attempt {attempt}/{MAX_DTYPE_RETRIES})...")
            try:
                if adapter.hybrid_step2(model_file, data_file, cfg_file):
                    step2_success = True
                    break
                else:
                    # Non-zero return but no exception — check if it's a dtype issue
                    # RKNN sometimes prints the error to stdout/stderr rather than raising
                    logger.warning(f"⚠️ Hybrid Step 2 returned non-zero on attempt {attempt}")
                    break  # Cannot parse error from return code alone
            except Exception as e:
                error_str = str(e)
                tb_str = traceback.format_exc()
                logger.warning(f"⚠️ Hybrid Step 2 exception on attempt {attempt}: {error_str}")

                # RKNN actual format:
                #   "The quantize_parameters['TENSOR_NAME']['dtype'] is not allowed to be modified!"
                combined_err = error_str + "\n" + tb_str
                dtype_match = re.search(
                    r"quantize_parameters\['([^']+)'\]\['dtype'\]\s+is\s+not\s+allowed\s+to\s+be\s+modified",
                    combined_err
                )

                if dtype_match:
                    offending_tensor = dtype_match.group(1)

                    # Guard: if we already removed this tensor and it still fails,
                    # the removal didn't work — abort instead of looping forever
                    if offending_tensor in removed_tensors:
                        logger.error(
                            f"   ❌ Tensor {offending_tensor} was already removed but still causes error. "
                            f"Possible cfg format mismatch. Aborting."
                        )
                        break

                    logger.info(
                        f"   🔧 Detected immutable tensor: {offending_tensor}. "
                        f"Removing its entry entirely from {cfg_file} and retrying..."
                    )
                    removed = self._remove_tensor_from_cfg(cfg_file, offending_tensor)
                    if not removed:
                        logger.error(f"   ❌ Failed to remove tensor {offending_tensor} from cfg. Aborting.")
                        break

                    removed_tensors.add(offending_tensor)
                    # Need a fresh adapter for retry — release old one, create new one
                    adapter.release()
                    adapter = RKNNAdapter(target_plat, verbose=True)
                    adapter.config(base_build_config, custom_string)
                    if not adapter.load_onnx(onnx_path, input_shapes):
                        logger.error("Failed to reload ONNX for retry.")
                        break
                    continue
                else:
                    # Check for RKNN SDK internal KeyError bug:
                    # In RKNN Toolkit 2.3.2, _p_adjust_tanh_sigmoid uses a key without
                    # the '_mm' suffix that Step1 generates for SE block MatMul outputs.
                    # This is an SDK bug — no workaround exists on our side.
                    if "KeyError" in tb_str and "_p_adjust_tanh_sigmoid" in tb_str:
                        key_match = re.search(r"KeyError:\s*'([^']+)'", tb_str)
                        missing_key = key_match.group(1) if key_match else "unknown"
                        logger.warning(
                            f"⚠️  Hybrid quantization is not supported for this model on RKNN Toolkit 2.3.2.\n"
                            f"   Root cause: SDK internal KeyError in _p_adjust_tanh_sigmoid ('{missing_key}').\n"
                            f"   This is a known SDK bug affecting SE Block / MatMul→Sigmoid patterns.\n"
                            f"   ➡  Plain INT8 model is unaffected and has already been exported."
                        )
                    else:
                        logger.error(f"❌ Unrecoverable error in Hybrid Step 2:\n{tb_str}")
                    break

        if step2_success:
            # 5. Export
            if adapter.export(output_path):
                logger.info(f"✅ Hybrid Model successfully saved to {output_path}")
            else:
                logger.error("❌ Export failed after hybrid build.")
        else:
            logger.error(f"❌ Hybrid Step 2 failed after {attempt} attempt(s).")

        # Cleanup
        adapter.release()

        return

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------
    @staticmethod
    def _remove_tensor_from_cfg(cfg_path, tensor_name):
        """
        Completely removes a tensor's entry block from the .quantization.cfg file.

        RKNN raises "dtype is not allowed to be modified" not only when changing
        to float16, but whenever that tensor's entry exists at all in the cfg —
        meaning even writing back the original dtype still fails. The only fix is
        to delete the entire block so RKNN falls back to the default from the
        .model file (which is already correct).

        Args:
            cfg_path: Path to the .quantization.cfg file
            tensor_name: Name of the tensor block to remove entirely

        Returns:
            bool: True if found and removed, False otherwise
        """
        try:
            with open(cfg_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            found = False
            i = 0

            while i < len(lines):
                line = lines[i]
                parts = line.split(':')

                if len(parts) >= 2:
                    current_key = parts[0].strip().strip('"').strip("'")

                    if current_key == tensor_name:
                        found = True
                        # Record indentation of this header line
                        layer_indent = len(line) - len(line.lstrip())
                        i += 1

                        # Skip all child lines that belong to this block
                        while i < len(lines):
                            current_line = lines[i]
                            current_indent = len(current_line) - len(current_line.lstrip())
                            if current_line.strip() and current_indent <= layer_indent:
                                break
                            i += 1

                        logger.info(f"   ✅ Removed immutable tensor entry: {tensor_name}")
                        continue  # Do NOT append — skip the whole block

                new_lines.append(line)
                i += 1

            if found:
                with open(cfg_path, 'w') as f:
                    f.writelines(new_lines)
                return True
            else:
                logger.warning(f"   ⚠️ Tensor {tensor_name} not found in {cfg_path}")
                return False

        except Exception as e:
            logger.error(f"   ❌ Error removing tensor {tensor_name} from cfg: {e}")
            return False
