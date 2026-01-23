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

from rknn.api import RKNN
from core.utils import logger
import os
import json
import re
from core.quantization.analyzer import QuantizationAnalyzer


class RKNNAdapter:
    """
    Decoupled interface for Rockchip RKNN Toolkit2.
    """

    def __init__(self, target_platform, verbose=False):
        self.target = target_platform
        self.verbose = verbose
        # Initialize RKNN instance
        self.rknn = RKNN(verbose=self.verbose)
        logger.info(f"RKNN Toolkit initialized for target: {self.target}")

    def config(self, config_dict, custom_string=None):
        """
        Independent configuration method.
        """
        logger.info("--> Configuring RKNN...")

        rknn_config_args = {
            "target_platform": self.target,
            "optimization_level": config_dict.get('optimization_level', 3),
            "custom_string": custom_string,
        }

        if config_dict.get('pruning', False):
            rknn_config_args['model_pruning'] = True

        # Standard quantization setup
        quant_config = config_dict.get('quantization', {})
        if quant_config.get('enabled', False):
            # Default to what's in config, or fallback
            rknn_config_args['quantized_dtype'] = quant_config.get('dtype', 'asymmetric_quantized-8')

        logger.debug(f"Config Args: {rknn_config_args}")
        self.rknn.config(**rknn_config_args)

    def load_onnx(self, onnx_path, input_shapes):
        """
        Independent Load ONNX method.
        """
        logger.info(f"--> Loading ONNX: {onnx_path}")
        # Parse input shapes [[1,80,50]] -> [[1,80,50]] (already list of lists)
        ret = self.rknn.load_onnx(model=onnx_path, inputs=None, input_size_list=input_shapes)
        if ret != 0:
            logger.error("Load ONNX failed!")
            return False
        return True

    def export(self, output_path):
        """
        Independent Export method.
        """
        logger.info(f"--> Exporting to: {output_path}")
        ret = self.rknn.export_rknn(output_path)
        if ret != 0:
            logger.error("Export RKNN failed!")
            return False
        return True

    def convert(self, onnx_path, output_path, input_shapes, config_dict, custom_string=None):
        """
        Independent full conversion method.
        """

        # logger.debug(f"Config Args: {rknn_config_args}")
        # self.rknn.config(**rknn_config_args)
        # logger.info("-----------------------\n")

        # 1. Config (Call the new method)
        logger.info("--> (1/5). Configuring RKNN...")
        self.config(config_dict, custom_string)
        logger.info("-----------------------\n")

        # 2. Load
        logger.info(f"--> (2/5). Loading ONNX: {onnx_path}")
        if not self.load_onnx(onnx_path, input_shapes):
            return False
        logger.info("-----------------------\n")
        # logger.info(f"--> (2/5). Loading ONNX: {onnx_path}")
        # # Parse input shapes [[1,80,50]] -> [[1,80,50]] (already list of lists)
        # load_ret = self.rknn.load_onnx(model=onnx_path, inputs=None, input_size_list=input_shapes)
        # if load_ret != 0:
        #     logger.error("Load ONNX failed!")
        #     return False
        # logger.info("-----------------------\n")

        # 3. Build
        logger.info("--> (3/5). Building RKNN Model...")
        do_quant = config_dict.get('quantization', {}).get('enabled', False)
        dataset = config_dict.get('quantization', {}).get('dataset', None)

        build_ret = self.rknn.build(do_quantization=do_quant, dataset=dataset)
        if build_ret != 0:
            logger.error("Build RKNN failed!")
            return False

        logger.info("-----------------------\n")

        # 4. Export
        logger.info(f"--> (4/5). Exporting to: {output_path}")
        if not self.export(output_path):
            return False
        logger.info("-----------------------\n")
        # logger.info(f"--> (4/5). Exporting to: {output_path}")
        # export_ret = self.rknn.export_rknn(output_path)
        # if export_ret != 0:
        #     logger.error("Export RKNN failed!")
        #     return False
        # logger.info("-----------------------\n")

        # 5. Evaluate (Memory)
        if config_dict.get('eval_memory', False):
            logger.info("--> (5/5). Evaluating Memory Usage...")
            self.rknn.init_runtime(target=self.target, eval_mem=True)
            mem_info = self.rknn.eval_memory()
            logger.info(f"Memory Profile:\n{mem_info}")
            logger.info("-----------------------\n")
        else:
            logger.info("--> (5/5). Skipping Memory Evaluation as per config.")
            logger.info("-----------------------\n")

        # ----------------------
        # release timing will be handled outside to allow further analysis if needed
        # self.rknn.release()
        return True

    def run_deep_analysis(self, dataset_path, output_dir):
        """
        Trigger deep accuracy analysis (layer-by-layer).
        This is a time-consuming operation.
        """
        logger.info("🩺 Triggering Deep Accuracy Analysis...")

        if not dataset_path or not os.path.exists(dataset_path):
            logger.error(f"Cannot run analysis: Dataset list not found at {dataset_path}")
            return

        try:
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # [Fix] Manually load the first sample from the dataset list
            # RKNN accuracy_analysis sometimes fails to parse complex txt lists with multiple inputs per line.
            # We load the .npy files of the first line manually to ensure safety.
            import numpy as np

            with open(dataset_path, 'r') as f:
                first_line = f.readline().strip()

            if not first_line:
                logger.error("Dataset list is empty!")
                return

            # Split by space to get paths for all inputs (feature + states)
            npy_paths = first_line.split()
            input_data_list = []

            for p in npy_paths:
                if not os.path.exists(p):
                    logger.error(f"Input npy file not found: {p}")
                    return
                input_data_list.append(np.load(p))

            logger.info(f"   Loaded {len(input_data_list)} input tensors for analysis.")

            # Execute analysis (target=None forces simulator mode)
            self.rknn.accuracy_analysis(inputs=input_data_list,
                                        output_dir=output_dir,
                                        target=None,
                                        device_id=None)
            logger.warning(f"⚠️  Analysis Report Generated: {output_dir}/error_analysis.txt")
            logger.warning(f"⚠️  Please check the report to identify layer-wise precision loss.")

        except Exception as e:
            logger.error(f"Accuracy Analysis crashed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def release(self):
        """Explicitly release RKNN resources."""
        if hasattr(self, 'rknn') and self.rknn:
            self.rknn.release()

    def hybrid_step1(self, dataset, proposal=False):
        """
        Wrapper for hybrid_quantization_step1.
        Generates .model, .data, and .quantization.cfg files.
        """
        logger.info("--> [Hybrid] Step (1/2): Generating intermediate files...")
        # rknn_batch_size=1 is required for this step usually
        ret = self.rknn.hybrid_quantization_step1(dataset=dataset, rknn_batch_size=1, proposal=proposal)
        return ret == 0

    def hybrid_step2(self, model_inp, data_inp, cfg_inp):
        """
        Wrapper for hybrid_quantization_step2.
        Generates the final quantized model in memory (needs export later? No, usually it saves to internal graph).
        Actually per SDK, this builds the model. We still need to call export_rknn afterwards?
        Wait, SDK says it generates "RKNN model".
        Usually standard flow is: step2 -> export_rknn.
        """
        logger.info("--> [Hybrid] Step (2/2): Building hybrid model...")
        ret = self.rknn.hybrid_quantization_step2(model_input=model_inp,
                                                  data_input=data_inp,
                                                  model_quantization_cfg=cfg_inp)
        return ret == 0

    def apply_hybrid_patch(self, cfg_path, analysis_path, threshold=0.99):
        """
        Reads the .quantization.cfg file generated by Step 1,
        Parses the error_analysis.txt to find layers with accuracy < threshold,
        Modifies the .cfg file to set those layers to 'float16'.
        """

        # Preparation -- 1. Echo welcome info
        logger.info(f"🔧 Patching quantization config based on analysis (Threshold: {threshold})...")

        # Preparation -- 2. Check config file existence
        if not os.path.exists(cfg_path):
            logger.error("Missing config file.")
            return False
        else:
            logger.info(f"   ✅[FOUND] {cfg_path}")

        # Preparation -- 3. Check analysis report existence
        if not os.path.exists(analysis_path):
            logger.error("Missing analysis report.")
            return False
        else:
            logger.info(f"   ✅[FOUND] {analysis_path}")

        # Preparation -- 4. Define ALLOWED types (Whitelist Mode)
        # Due to RKNN's limitations on many operators, we only allow modifications on those that are 100% safe and have the most impact on accuracy.
        # Usually: Conv, Gemm, MatMul, ConvTranspose.
        ALLOWED_TYPES = {
            'Conv',  # (Most common) Convolution
            'Gemm',  # Fully Connected Layer
            'MatMul',  # Matrix Multiplication
            'ConvTranspose',  # Anti-convolution
            'Linear',  # Some older converters may call it Linear
            # Note: Here we deliberately exclude Add, Mul, Div, because they are prone to fusion issues causing crashes
        }

        # Preparation -- 5. Initialize bad layer account
        bad_layer_account = 0

        # Processing -- 1. Parse Analysis Report to find bad layers
        bad_layers = set()
        # Matches: [Type] LayerName ... SingleCos
        # Log format: [Conv] 123_rs ... 0.999 ... 0.850
        # We need a robust regex similar to what we discussed
        # pattern = re.compile(r'^\[.*?\]\s+(\S+)\s+[\d\.]+\s+\|\s+[\d\.]+\s+([\d\.]+)')
        pattern = re.compile(r'^\[(.*?)\]\s+(\S+)\s+[0-9eE\.\-\+]+\s+\|\s+[0-9eE\.\-\+]+\s+([0-9eE\.\-\+]+)')

        with open(analysis_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not self._match_line_valid(line):
                    continue

                logger.info(f"Checking line: {line[:50]}...")
                match = pattern.match(line)

                # [Point 2]: Add diagnostic log for failed match
                if not match:
                    # If the line contains a vertical bar '|' but no match, it means the regex is wrong, must print it out
                    if '|' in line:
                        logger.warning(f"⚠️ [REGEX FAIL] Ignored line: {line}")
                    continue

                layer_type = match.group(1)
                layer_name = match.group(2)

                # [Point 3]: Add diagnostic log for score parsing
                try:
                    score = float(match.group(3))
                except ValueError:
                    logger.error(f"❌ [NUM ERROR] Cannot parse score: {match.group(3)}")
                    continue

                # [Point 3.5]: Safe Name Check
                # With '#' in layer_name, it is usually a derived node generated by Split/Slice, RKNN locks its precision
                if '#' in layer_name:
                    # if score < threshold:
                    logger.debug(
                        f"   [SKIP] {layer_name} [{layer_type}] (Score: {score:.4f}) -> Unsafe Internal Node (Contains '#')"
                    )
                    continue

                # [Point 4]: Decision process (Whitelist Mode)
                if layer_type not in ALLOWED_TYPES:
                    # if score < threshold:
                    logger.debug(
                        f"   [SKIP] {layer_name} [{layer_type}] (Score: {score:.4f}) -> Not in Allowed List")
                    continue

                # If it is an allowed type and the score is low, add it to the patch list
                if score < threshold:
                    # logger.info(f"   [ADD ] {layer_name} [{layer_type}] (Score: {score:.4f}) -> ✅ Added to Patch List")
                    bad_layers.add(layer_name)
                    bad_layer_account += 1
                    logger.info(
                        f"   📉 [No.{bad_layer_account}]: Found sensitive layer {layer_name} {layer_type} (Score: {score:.4f})"
                    )
                else:
                    # Score is high enough, no need to change
                    pass

        if not bad_layers:
            logger.info("   ✨ No layers found below threshold. No changes made.")
            return True
        else:
            logger.info(f"   ✅ [FOUND] {len(bad_layers)} sensitive layers below threshold")

        # Processing -- 2. Modify the .cfg file
        # Format in cfg: layer_name: quantized_dtype
        # e.g., "7206-rs: asymmetric_quantized-8"
        new_lines = []
        modified_count = 0
        i = 0

        with open(cfg_path, 'r') as f:
            lines = f.readlines()

        while i < len(lines):
            line = lines[i]

            # Check if this line starts with a bad layer name
            # Format usually: "layer_name: type"
            parts = line.split(':')

            if len(parts) >= 2:
                # ------
                # key = parts[0].strip()
                # ------
                # The original key might be quoted or not.
                # strip() removes whitespace and potentially quotes if we are not careful,
                # but split(':') is crude.

                # Robust approach: check if any bad layer name appears in the line
                # This handles "layer_name": dtype and layer_name: dtype

                # Let's simplify:
                # If we find a line starting with one of our bad layers (allowing for quotes), replace it.

                current_key = parts[0].strip().strip('"').strip("'")
                # ------

                if current_key in bad_layers:

                    # Preserve the layer name line, do not replace
                    new_lines.append(line)
                    i += 1

                    # Record the indentation of this layer line to know when we exit this block
                    layer_indent = len(line) - len(line.lstrip())
                    dtype_found = False

                    # Continue reading subsequent lines of this layer until we find dtype or exit this layer
                    while i < len(lines):
                        current_line = lines[i]
                        current_indent = len(current_line) - len(current_line.lstrip())

                        # If the indentation is the same or less, we have exited this layer block
                        if current_line.strip() and current_indent <= layer_indent:
                            break

                        # Find the dtype line and modify it
                        if current_line.strip().startswith('dtype:'):
                            indent = current_line[:len(current_line) - len(current_line.lstrip())]
                            new_lines.append(f'{indent}dtype: float16\n')
                            dtype_found = True
                            modified_count += 1
                            i += 1
                            continue

                        # Keep other lines unchanged
                        new_lines.append(current_line)
                        i += 1

                    if not dtype_found:
                        logger.warning(f"   ⚠️  Could not find dtype field for layer {current_key}")
                    continue

            new_lines.append(line)
            i += 1

        # 3. Save back
        with open(cfg_path, 'w') as f:
            f.writelines(new_lines)

        logger.info(f"   ✅ Patched {modified_count} layers to float16 in {cfg_path}")
        return True

    def _match_line_valid(self, line):
        if not line or \
            line.startswith('#') or \
            line.startswith('-') or \
                "layer_name" in line:
            return False
        else:
            return True
