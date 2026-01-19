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

        # # 1. Config
        # logger.info("--> (1/5). Configuring RKNN...")
        # # Map YAML config keys to rknn.config arguments
        # # Note: 'target_platform' in rknn.config expects lowercase, e.g., 'rv1126'
        # # The SDK user might pass 'rv1126b', we pass it as is, assuming toolkit handles it or user configured correctly.

        # rknn_config_args = {
        #     "target_platform": self.target,
        #     "optimization_level": config_dict.get('optimization_level', 3),
        #     "custom_string": custom_string,
        #     # Add other config mapping here if needed
        # }

        # if config_dict.get('pruning', False):
        #     rknn_config_args['model_pruning'] = True

        # # Quantization setup
        # quant_config = config_dict.get('quantization', {})
        # if quant_config.get('enabled', False):
        #     rknn_config_args['quantized_dtype'] = quant_config['dtype']

        #     # # === [修改点] 加载混合量化配置 JSON 为字典 ===
        #     # hybrid_conf_path = config_dict.get('quantization', {}).get('hybrid_config_path')
        #     # if hybrid_conf_path and os.path.exists(hybrid_conf_path):
        #     #     logger.info(f"⚡ Hybrid Quantization Enabled! Loading config from: {hybrid_conf_path}")
        #     #     try:
        #     #         import json
        #     #         with open(hybrid_conf_path, 'r') as f:
        #     #             quant_config_dict = json.load(f)

        #     #         # 这里的参数名根据 SDK 版本可能不同，Toolkit2 常用 'quantization_config' 或直接合并
        #     #         # 通常 safe 的做法是直接传给 config
        #     #         rknn_config_args['quantization_config'] = quant_config_dict
        #     #     except Exception as e:
        #     #         logger.error(f"Failed to load hybrid config: {e}")
        #     # # ==========================================

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

    # def generate_quant_config(self, analysis_report_path, output_config_path, auto_threshold=None):
    #     """
    #     Parses the error_analysis.txt and generates a hybrid quantization config.

    #     Args:
    #         analysis_report_path (str): Path to the RKNN accuracy analysis txt.
    #         output_config_path (str): Path where the JSON config will be saved.
    #         auto_threshold (float, optional):
    #             If provided (e.g., 0.99), layers with single-layer cosine similarity
    #             below this value will be set to 'float16'.
    #             If None, all layers are set to 'int8' for manual editing.
    #     """

    #     # Preparation -- 1. Echo welcome info
    #     logger.info(f"📝 Generating quantization config template from analysis report...")

    #     # Preparation -- 2. Check report existence
    #     if not os.path.exists(analysis_report_path):
    #         logger.error(f"Analysis report not found at {analysis_report_path}")
    #         return False

    #     # Preparation -- 3. Determine hybrid-quantization mode (Manual or Auto)
    #     use_auto_mode = auto_threshold is not None
    #     if use_auto_mode:
    #         logger.info(f"   Mode: AUTO (Threshold: {auto_threshold})")
    #     else:
    #         logger.info(f"   Mode: MANUAL template generation")

    #     # Processing -- 4. Define data structures
    #     layer_configs = {}
    #     # Regex to capture: [Type] LayerName ... EntireCos | EntireEuc SingleCos ...
    #     # Based on log: [Reshape] cached_conv1_0_rs  1.00000 | 0.0  0.99000 | 0.0
    #     # We look for the pattern and specifically the 3rd number (Single Cosine).

    #     # Pattern explanation:
    #     # ^\[.*?\]\s+   : Start with [Type] and spaces
    #     # (\S+)         : Capture Group 1: Layer Name (non-whitespace)
    #     # \s+           : Spaces
    #     # [\d\.]+\s+\|\s+[\d\.]+ : Skip Entire Cos | Entire Euc
    #     # \s+           : Spaces
    #     # ([\d\.]+)     : Capture Group 2: Single Cosine Score
    #     line_pattern = re.compile(r'^\[.*?\]\s+(\S+)\s+[\d\.]+\s+\|\s+[\d\.]+\s+([\d\.]+)')

    #     # Processing -- 4. Parse the report
    #     try:
    #         with open(analysis_report_path, 'r') as f:
    #             for line in f:
    #                 # Logic -- a. Strip line
    #                 line = line.strip()

    #                 # Logic -- b. Skip non-layer lines
    #                 if  not line or \
    #                     line.startswith('#') or \
    #                     line.startswith('-') or \
    #                     "layer_name" in line:
    #                     continue

    #                 # Logic -- c. Extract layer name
    #                 # 匹配: [Conv] 7206-rs ...
    #                 # 提取 [] 后面的第一个单词作为层名
    #                 match = line_pattern.match(line)

    #                 # Logic -- d. Default to int8 for all layers found
    #                 if match:
    #                     layer_name = match.group(1)
    #                     single_cosine_str = match.group(2)

    #                     # Logic -- e. Determine cosine score
    #                     try:
    #                         single_cosine = float(single_cosine_str)
    #                     except ValueError:
    #                         logger.warning(
    #                             f"   Could not parse cosine score for layer {layer_name}, skipping...")
    #                         single_cosine = 1.0  # Default to safe value
    #                         continue

    #                     # Logic -- f. Decide layer dtype based on mode
    #                     if use_auto_mode:
    #                         # Auto Mode: If score is bad, use float16. Otherwise keep int8 defaults (or empty)
    #                         # To be safe, we only write the overridden layers to the config.
    #                         if single_cosine < auto_threshold:
    #                             logger.debug(
    #                                 f"   📉 Layer {layer_name} score {single_cosine:.4f} < {auto_threshold}. Set to float16."
    #                             )
    #                             layer_configs[layer_name] = "float16"
    #                         else:
    #                             # For auto mode, we usually don't need to explicitly set int8
    #                             # unless we want to lock it. RKNN defaults to int8.
    #                             # Let's skip writing good layers to keep config clean,
    #                             # or write them as int8 if strict control is needed.
    #                             pass
    #                     else:
    #                         # Manual Mode: Dump everything as int8 so user can see and edit.
    #                         layer_configs[layer_name] = "int8"

    #         # If Auto mode found no bad layers, but the global score was low,
    #         # it might be an accumulation error.
    #         if use_auto_mode and not layer_configs:
    #             logger.warning(
    #                 "   [Auto] No single layer dropped below threshold. Problem might be cumulative.")

    #         # Toolkit2 的混合量化配置通常是一个字典，键是层名，值是精度
    #         # 有时需要包裹在 'override_layer_configs' 或直接作为 config
    #         # 根据经验，Toolkit2 接受直接的层名映射，或者需要查阅具体版本的 manual
    #         # 这里我们生成最通用的 {layer: dtype} 格式

    #         # Logic -- 6. Write to JSON
    #         with open(output_config_path, 'w') as f:
    #             json.dump(layer_configs, f, indent=4)

    #         # Logic -- 7. Return success
    #         return True
    #     except Exception as e:
    #         logger.error(f"Failed to generate config from report: {e}")
    #         import traceback
    #         logger.error(traceback.format_exc())
    #         return False

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
        ret = self.rknn.hybrid_quantization_step1(
            dataset=dataset,
            rknn_batch_size=1,
            proposal=proposal
        )
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
        ret = self.rknn.hybrid_quantization_step2(
            model_input=model_inp,
            data_input=data_inp,
            model_quantization_cfg=cfg_inp
        )
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
        if not os.path.exists(cfg_path) :
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

        # Preparation -- 4. Define ignored types
        # [Blacklist] These layers are structural layers, RKNN prohibits users from modifying their dtype
        # Even if their scores are low, they cannot be moved, otherwise Step 2 will report an error
        IGNORED_TYPES = {
            'Input', 'Output', 'DataConvert',
            'Reshape', 'Transpose', 'Permute', 'Flatten',
            'Concat', 'Split', 'Slice', 'Gather', 'Resize', 'Pad', 'Clip',
            'Softmax', 'Sigmoid' # 某些版本的 RKNN 对激活函数层也有限制，建议先加上
        }

        # Preparation -- 5. Initialize bad layer account
        bad_layer_account = 0

        # Processing -- 1. Parse Analysis Report to find bad layers
        bad_layers = set()
        # Matches: [Type] LayerName ... SingleCos
        # Log format: [Conv] 123_rs ... 0.999 ... 0.850
        # We need a robust regex similar to what we discussed
        pattern = re.compile(r'^\[.*?\]\s+(\S+)\s+[\d\.]+\s+\|\s+[\d\.]+\s+([\d\.]+)')

        with open(analysis_path, 'r') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    layer_type = match.group(1)
                    layer_name = match.group(2)

                    if layer_type in IGNORED_TYPES:
                        continue

                    try:
                        score = float(match.group(3))
                        if score < threshold:
                            bad_layers.add(layer_name)
                            bad_layer_account += 1
                            logger.debug(f"   📉 Found sensitive layer - {bad_layer_account}: {layer_name} {layer_type} (Score: {score:.4f})")
                    except:
                        pass

        if not bad_layers:
            logger.info("   ✨ No layers found below threshold. No changes made.")
            return True

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
                    # 保留层名行，不替换
                    new_lines.append(line)
                    i += 1

                    # 记录层名的缩进，用于判断何时离开这个层的块
                    layer_indent = len(line) - len(line.lstrip())
                    dtype_found = False

                    # 继续读取这个层的后续行，直到找到 dtype 或离开这个层
                    while i < len(lines):
                        current_line = lines[i]
                        current_indent = len(current_line) - len(current_line.lstrip())

                        # 如果缩进相同或更少，说明已经离开这个层的块了
                        if current_line.strip() and current_indent <= layer_indent:
                            break

                        # 找到 dtype 行，修改它
                        if current_line.strip().startswith('dtype:'):
                            indent = current_line[:len(current_line) - len(current_line.lstrip())]
                            new_lines.append(f'{indent}dtype: float16\n')
                            dtype_found = True
                            modified_count += 1
                            i += 1
                            continue

                        # 其他行原样保留
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
