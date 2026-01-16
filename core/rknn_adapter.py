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

    def convert(self, onnx_path, output_path, input_shapes, config_dict, custom_string=None):

        # 1. Config
        logger.info("--> (1/5). Configuring RKNN...")
        # Map YAML config keys to rknn.config arguments
        # Note: 'target_platform' in rknn.config expects lowercase, e.g., 'rv1126'
        # The SDK user might pass 'rv1126b', we pass it as is, assuming toolkit handles it or user configured correctly.

        rknn_config_args = {
            "target_platform": self.target,
            "optimization_level": config_dict.get('optimization_level', 3),
            "custom_string": custom_string,
            # Add other config mapping here if needed
        }

        if config_dict.get('pruning', False):
            rknn_config_args['model_pruning'] = True

        # Quantization setup
        quant_config = config_dict.get('quantization', {})
        if quant_config.get('enabled', False):
            rknn_config_args['quantized_dtype'] = quant_config['dtype']

            # === [修改点] 加载混合量化配置 JSON 为字典 ===
            hybrid_conf_path = config_dict.get('quantization', {}).get('hybrid_config_path')
            if hybrid_conf_path and os.path.exists(hybrid_conf_path):
                logger.info(
                    f"⚡ Hybrid Quantization Enabled! Loading config from: {hybrid_conf_path}")
                try:
                    import json
                    with open(hybrid_conf_path, 'r') as f:
                        quant_config_dict = json.load(f)

                    # 这里的参数名根据 SDK 版本可能不同，Toolkit2 常用 'quantization_config' 或直接合并
                    # 通常 safe 的做法是直接传给 config
                    rknn_config_args['quantization_config'] = quant_config_dict
                except Exception as e:
                    logger.error(f"Failed to load hybrid config: {e}")
            # ==========================================

        logger.debug(f"Config Args: {rknn_config_args}")
        self.rknn.config(**rknn_config_args)
        logger.info("-----------------------\n")

        # 2. Load
        logger.info(f"--> (2/5). Loading ONNX: {onnx_path}")
        # Parse input shapes [[1,80,50]] -> [[1,80,50]] (already list of lists)
        load_ret = self.rknn.load_onnx(model=onnx_path, inputs=None, input_size_list=input_shapes)
        if load_ret != 0:
            logger.error("Load ONNX failed!")
            return False
        logger.info("-----------------------\n")

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
        export_ret = self.rknn.export_rknn(output_path)
        if export_ret != 0:
            logger.error("Export RKNN failed!")
            return False
        logger.info("-----------------------\n")

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

    def generate_quant_config(self, analysis_report_path, output_config_path):
        """
        解析 error_analysis.txt 生成混合量化配置文件模板 (JSON)
        """
        import json
        import re

        logger.info(f"📝 Generating quantization config template from analysis report...")

        if not os.path.exists(analysis_report_path):
            logger.error(f"Analysis report not found at {analysis_report_path}")
            return False

        # 默认模板结构
        quant_config = {
            # "quantized_dtype": "asymmetric_quantized-8", # 可选，全局配置
            # "quantized_algorithm": "normal",
        }

        # 你的 error_analysis.txt 格式: [Type] LayerName ...
        # 我们需要提取 LayerName
        layer_dtypes = {}

        try:
            with open(analysis_report_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # 跳过注释和空行
                    if not line or line.startswith('#') or line.startswith(
                            '-') or "layer_name" in line:
                        continue

                    # 匹配: [Conv] 7206-rs ...
                    # 提取 [] 后面的第一个单词作为层名
                    match = re.match(r'^\[.*?\]\s+(\S+)', line)
                    if match:
                        layer_name = match.group(1)
                        # 默认为 int8，用户将需要修改的改为 float16
                        layer_dtypes[layer_name] = "int8"

            if not layer_dtypes:
                logger.warning("No layers found in analysis report. Check parsing logic.")
                return False

            # Toolkit2 的混合量化配置通常是一个字典，键是层名，值是精度
            # 有时需要包裹在 'override_layer_configs' 或直接作为 config
            # 根据经验，Toolkit2 接受直接的层名映射，或者需要查阅具体版本的 manual
            # 这里我们生成最通用的 {layer: dtype} 格式

            with open(output_config_path, 'w') as f:
                json.dump(layer_dtypes, f, indent=4)

            return True
        except Exception as e:
            logger.error(f"Failed to generate config from report: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

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
