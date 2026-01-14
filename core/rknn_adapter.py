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

    def convert(self, 
                onnx_path, 
                output_path, 
                input_shapes, 
                config_dict, 
                custom_string=None):
        
        # 1. Config
        logger.info("--> Configuring RKNN...")
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
             
        # Quantization type mapping
        if config_dict.get('quantization', {}).get('enabled', False):
             rknn_config_args['quantized_dtype'] = config_dict['quantization']['dtype']
        
        logger.debug(f"Config Args: {rknn_config_args}")
        self.rknn.config(**rknn_config_args)

        # 2. Load
        logger.info(f"--> Loading ONNX: {onnx_path}")
        # Parse input shapes [[1,80,50]] -> [[1,80,50]] (already list of lists)
        load_ret = self.rknn.load_onnx(model=onnx_path, inputs=None, input_size_list=input_shapes)
        if load_ret != 0:
            logger.error("Load ONNX failed!")
            return False

        # 3. Build
        logger.info("--> Building RKNN Model...")
        do_quant = config_dict.get('quantization', {}).get('enabled', False)
        dataset = config_dict.get('quantization', {}).get('dataset', None)
        
        build_ret = self.rknn.build(do_quantization=do_quant, dataset=dataset)
        if build_ret != 0:
            logger.error("Build RKNN failed!")
            return False

        # 4. Export
        logger.info(f"--> Exporting to: {output_path}")
        export_ret = self.rknn.export_rknn(output_path)
        if export_ret != 0:
            logger.error("Export RKNN failed!")
            return False

        # 5. Evaluate (Memory)
        if config_dict.get('eval_memory', False):
            logger.info("--> Evaluating Memory Usage...")
            self.rknn.init_runtime(target=self.target, eval_mem=True)
            mem_info = self.rknn.eval_memory()
            logger.info(f"Memory Profile:\n{mem_info}")

        self.rknn.release()
        return True
