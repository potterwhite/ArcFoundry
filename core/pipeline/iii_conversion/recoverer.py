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
import importlib.metadata
from utils import logger, timed_input
from ._rknn_adapter import RKNNAdapter


def _get_rknn_toolkit_version():
    """Dynamically detect the installed rknn-toolkit2 version at runtime."""
    for pkg in ("rknn-toolkit2", "rknn_toolkit2", "rknn-toolkit-lite2"):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


class PrecisionRecoverer:

    def __init__(self, global_config):
        self.cfg = global_config
        self.output_dir = self.cfg.get("project", {}).get("output_dir", "./output")

    # --------------------------------------------------------------------------
    # Public Entry Point
    # --------------------------------------------------------------------------
    def _recover_precision(self, target_plat, model_name, onnx_path, output_path,
                           input_shapes, base_build_config, custom_string):
        """
        Hybrid Quantization Workflow orchestrator.
        Delegates each phase to a focused sub-method:
          1. Run hybrid Step 1 (generate .model / .data / .quantization.cfg)
          2. Select patch strategy (auto or manual)
          3. Run hybrid Step 2 with dtype-error retry
          4. Export on success
        """
        logger.info(f"🚑 Entering Accuracy Recovery Workflow for {model_name}...")

        paths = self._prepare_hybrid_paths(onnx_path, model_name)

        adapter = self._run_hybrid_step1(
            target_plat, onnx_path, input_shapes, base_build_config, custom_string)
        if adapter is None:
            return

        self._select_patch_strategy(adapter, paths['cfg'], paths['error_report'])

        result, adapter = self._run_hybrid_step2_with_retry(
            adapter, target_plat, onnx_path, input_shapes,
            base_build_config, custom_string,
            paths['cfg'], paths['model'], paths['data'])

        if result == 'success':
            if adapter.export(output_path):
                logger.info(f"✅ Hybrid Model successfully saved to {output_path}")
            else:
                logger.error("❌ Export failed after hybrid build.")
        elif result == 'sdk_limitation':
            logger.info("ℹ️  Hybrid quantization skipped (SDK limitation). Plain INT8 model stands.")
        else:
            logger.error(f"❌ Hybrid quantization failed: {result}")

        if adapter is not None:
            adapter.release()

    # --------------------------------------------------------------------------
    # Sub-step Methods
    # --------------------------------------------------------------------------
    def _prepare_hybrid_paths(self, onnx_path, model_name):
        """Return a dict of all file paths used by the hybrid workflow."""
        onnx_basename = os.path.basename(onnx_path)
        model_prefix = os.path.splitext(onnx_basename)[0]
        analysis_dir = os.path.join(self.output_dir, "analysis", model_name)
        return {
            'cfg':          f"{model_prefix}.quantization.cfg",
            'model':        f"{model_prefix}.model",
            'data':         f"{model_prefix}.data",
            'error_report': os.path.join(analysis_dir, "error_analysis.txt"),
        }

    def _run_hybrid_step1(self, target_plat, onnx_path, input_shapes,
                          base_build_config, custom_string):
        """
        Prompt user to confirm, then run hybrid_quantization_step1.
        Returns a configured RKNNAdapter on success, None on skip or failure.
        """
        logger.info(f"\n[INTERVENTION] Accuracy is below threshold.")
        choice = timed_input("   >>> Start Hybrid Quantization Step 1/2? [Y/n]: ",
                             timeout=15, default='y')
        if choice not in ('', 'y', 'yes'):
            return None

        logger.info(f"\n\n   🔄 Starting Hybrid Quantization Step 1/2...")
        adapter = self._make_fresh_adapter(
            target_plat, onnx_path, input_shapes, base_build_config, custom_string)
        if adapter is None:
            return None

        dataset_path = base_build_config.get('quantization', {}).get('dataset')
        if not adapter.hybrid_step1(dataset_path):
            logger.error("Hybrid Step 1/2 failed.")
            adapter.release()
            return None

        logger.info("   ✨ Step 1/2 Complete.")
        return adapter

    def _select_patch_strategy(self, adapter, cfg_file, error_report):
        """
        Prompt user to choose auto-patch or manual edit, then apply the patch.
        """
        logger.info("\n\n   ✨ Step 2/2 Started:")
        logger.info("   [SELECT STRATEGY]")
        logger.info("   (a) Auto-Patch: Automatically set layers < threshold to float16.")
        logger.info("   (m) Manual: You edit the .cfg file yourself.")
        mode = timed_input("   >>> Select mode [a/m] (default: a): ", timeout=10, default='a')

        if mode == 'm':
            logger.info(f"\n   !!! ACTION: Please edit ./{cfg_file} now.")
            logger.info("   Find sensitive layers and change 'asymmetric_quantized-8' to 'float16'.")
            input("   >>> Press [ENTER] when you are ready for Step 2...")
        else:
            thresh_input = timed_input(
                "   >>> Enter min cosine score threshold (default 0.99): ",
                timeout=15, default='0.99')
            try:
                threshold = float(thresh_input) if thresh_input else 0.99
            except ValueError:
                threshold = 0.99
            adapter.apply_hybrid_patch(cfg_file, error_report, threshold)

    def _run_hybrid_step2_with_retry(self, adapter, target_plat, onnx_path, input_shapes,
                                     base_build_config, custom_string,
                                     cfg_file, model_file, data_file):
        """
        Run hybrid_step2, retrying after each "dtype not allowed" by removing
        the offending tensor from cfg. Each retry handles one locked tensor.

        Returns:
            (result_str, adapter)
            result_str: 'success' | 'sdk_limitation' | 'failed: <reason>'
        """
        MAX_RETRIES = 10
        removed_tensors = set()

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(f"🔄 Executing Hybrid Step 2 (attempt {attempt}/{MAX_RETRIES})...")
            try:
                if adapter.hybrid_step2(model_file, data_file, cfg_file):
                    return 'success', adapter
                else:
                    logger.warning(f"⚠️ Hybrid Step 2 returned non-zero on attempt {attempt}")
                    return 'failed: non-zero return', adapter

            except Exception as e:
                action, adapter = self._handle_step2_exception(
                    e, traceback.format_exc(), cfg_file, removed_tensors,
                    adapter, target_plat, onnx_path, input_shapes,
                    base_build_config, custom_string)

                if action == 'retry':
                    continue
                return action, adapter

        return f'failed: exceeded {MAX_RETRIES} retries', adapter

    def _handle_step2_exception(self, exc, tb_str, cfg_file, removed_tensors,
                                adapter, target_plat, onnx_path, input_shapes,
                                base_build_config, custom_string):
        """
        Classify a hybrid_step2 exception and return (action, adapter).

        Actions:
          'retry'           — removed offending tensor, caller should retry
          'sdk_limitation'  — known SDK bug, abort gracefully (no error log)
          'failed: <why>'   — unrecoverable, caller should stop
        """
        combined = str(exc) + "\n" + tb_str
        logger.warning(f"⚠️ Hybrid Step 2 exception: {exc}")

        # Case 1: RKNN refuses to let us modify this tensor's dtype at all.
        # Fix: remove the entry entirely from cfg so RKNN uses its default.
        dtype_match = re.search(
            r"quantize_parameters\['([^']+)'\]\['dtype'\]\s+is\s+not\s+allowed\s+to\s+be\s+modified",
            combined
        )
        if dtype_match:
            return self._handle_dtype_error(
                dtype_match.group(1), cfg_file, removed_tensors,
                adapter, target_plat, onnx_path, input_shapes,
                base_build_config, custom_string)

        # Case 2: RKNN SDK KeyError in _p_adjust_tanh_sigmoid.
        # Root cause: SDK uses tensor key without '_mm' suffix that Step1 generates
        # for SE Block MatMul outputs. No workaround — abort gracefully.
        if "KeyError" in tb_str and "_p_adjust_tanh_sigmoid" in tb_str:
            return self._handle_sdk_se_block_bug(tb_str, adapter)

        # Case 3: Unknown — log full traceback
        logger.error(f"❌ Unrecoverable error in Hybrid Step 2:\n{tb_str}")
        return 'failed: unrecoverable exception', adapter

    def _handle_dtype_error(self, tensor_name, cfg_file, removed_tensors,
                            adapter, target_plat, onnx_path, input_shapes,
                            base_build_config, custom_string):
        """Remove an immutable tensor entry from cfg and prepare for retry."""
        if tensor_name in removed_tensors:
            logger.error(
                f"   ❌ Tensor '{tensor_name}' was already removed but still causes the same error. "
                f"Possible cfg format mismatch. Aborting."
            )
            return 'failed: removal ineffective', adapter

        logger.info(
            f"   🔧 Immutable tensor '{tensor_name}': removing entry from cfg and retrying...")
        if not self._remove_tensor_from_cfg(cfg_file, tensor_name):
            logger.error(f"   ❌ Failed to remove tensor '{tensor_name}' from cfg.")
            return 'failed: cfg removal error', adapter

        removed_tensors.add(tensor_name)
        adapter.release()
        adapter = self._make_fresh_adapter(
            target_plat, onnx_path, input_shapes, base_build_config, custom_string)
        if adapter is None:
            return 'failed: adapter reload error', None

        return 'retry', adapter

    def _handle_sdk_se_block_bug(self, tb_str, adapter):
        """
        Gracefully handle the RKNN SDK _p_adjust_tanh_sigmoid KeyError.
        This bug exists because the SDK's quant_optimizer looks up a tensor
        key without the '_mm' suffix that hybrid_step1 generates for SE Block
        MatMul outputs. It is not fixable on our side.
        """
        rknn_ver = _get_rknn_toolkit_version()
        key_match = re.search(r"KeyError:\s*'([^']+)'", tb_str)
        missing_key = key_match.group(1) if key_match else "unknown"
        logger.warning(
            f"⚠️  Hybrid quantization is not supported for this model "
            f"(rknn-toolkit2 {rknn_ver}).\n"
            f"   Root cause: SDK internal KeyError in _p_adjust_tanh_sigmoid "
            f"(missing key: '{missing_key}').\n"
            f"   Known SDK bug: SE Block MatMul→Sigmoid tensor key mismatch.\n"
            f"   ➡  Plain INT8 model is unaffected and has already been exported."
        )
        return 'sdk_limitation', adapter

    def _make_fresh_adapter(self, target_plat, onnx_path, input_shapes,
                            base_build_config, custom_string):
        """Create, configure, and load ONNX into a new RKNNAdapter. Returns None on failure."""
        adapter = RKNNAdapter(target_plat, verbose=True)
        adapter.config(base_build_config, custom_string)
        if not adapter.load_onnx(onnx_path, input_shapes):
            logger.error("Failed to load ONNX into adapter.")
            adapter.release()
            return None
        return adapter

    # --------------------------------------------------------------------------
    # Static Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _remove_tensor_from_cfg(cfg_path, tensor_name):
        """
        Completely removes a tensor's entry block from the .quantization.cfg file.

        RKNN raises "dtype is not allowed to be modified" whenever that tensor's
        entry exists in cfg — regardless of the dtype value written. The only fix
        is to delete the entire block so RKNN falls back to the default from the
        .model file.

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
                        layer_indent = len(line) - len(line.lstrip())
                        i += 1

                        # Skip all child lines belonging to this block
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
                logger.warning(f"   ⚠️ Tensor '{tensor_name}' not found in {cfg_path}")
                return False

        except Exception as e:
            logger.error(f"   ❌ Error removing tensor '{tensor_name}' from cfg: {e}")
            return False
