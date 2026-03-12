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

import onnx
import onnxsim
import onnxruntime
import os
#from core.quantization import strategies
from utils import logger
from .dynamic_shape_fixer import DynamicShapeFixer


class Preprocessor:
    """
    Handles all ONNX graph modifications before RKNN conversion.
    Strategies: Fix Dynamic Shape, Simplify, Extract Metadata, Fix Types.
    """

    def __init__(self, config):
        self.cfg = config

    def _fix_int64_type(self, model):
        """Forces input type to INT64 (Crucial for Sherpa Decoder)."""
        changed = False
        for input_tensor in model.graph.input:
            # Heuristic: If it's the specific 'y' input or generally needed
            # For now, apply to 'y' as seen in original script, or make generic via config later.
            if input_tensor.name == 'y':
                logger.debug(
                    f"  - Forcing input '{input_tensor.name}' to INT64")
                input_tensor.type.tensor_type.elem_type = onnx.TensorProto.INT64
                changed = True
        return changed

    def _simplify(self, input_path, output_path):
        """Wraps onnxsim."""
        try:
            logger.debug(f"\tinput_path={input_path}")
            logger.debug(f"\toutput_path={output_path}")
            model = onnx.load(input_path)
            model_simp, check = onnxsim.simplify(model)
            if check:
                onnx.save(model_simp, output_path)
                return output_path
            else:
                logger.warning(
                    "onnxsim check failed, using unsimplified model.")
                return input_path
        except Exception as e:
            logger.error(f"onnxsim failed: {e}")
            return input_path

    def _extract_metadata(self, onnx_path):
        """Extracts custom_metadata_map using ONNXRuntime."""
        try:
            # We use CPU provider to avoid any NPU dependency at this stage
            sess_options = onnxruntime.SessionOptions()
            # Suppress logs
            sess_options.log_severity_level = 3
            session = onnxruntime.InferenceSession(
                onnx_path, sess_options, providers=["CPUExecutionProvider"])

            meta = session.get_modelmeta()
            custom_map = meta.custom_metadata_map

            if not custom_map:
                logger.warning("No custom metadata found in ONNX.")
                return None

            # Format: key=value;key2=value2
            kv_string = ";".join([f"{k}={v}" for k, v in custom_map.items()])
            logger.info(f"  - Captured Metadata: {kv_string}")
            return kv_string

        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return None

    def preprocess(self, model_name, model_path, model_shapes, output_path,
                   json_strategies):
        """
        Applies a series of preprocessing strategies to the ONNX model.
        """
        # Preparation -- 1. Define Variables
        current_model_path = model_path
        custom_string = None
        is_model_modified = False

        # Processing -- 1. Extract Metadata (Sherpa specific - Must run on original ONNX)
        if json_strategies.get('extract_metadata', False):
            logger.info("Strategy (1/4): Extracting Custom Metadata...")
            custom_string = self._extract_metadata(current_model_path)
        else:
            logger.info("Strategy (1/4): Metadata Extraction Disabled.")

        # Preparation -- 2. Existing Check
        if os.path.exists(output_path):
            try:
                logger.info(f"[Cache Hit] Found existing file: {output_path}")
                onnx.load(output_path)
                logger.info(
                    "⏩ [FAST-FORWARD] Model is valid. Skipping complex preprocessing steps.\n"
                )

                return output_path, custom_string
            except Exception:
                logger.warning(
                    f"   Cached model corrupted or invalid ({e}). removing and regenerating..."
                )
                try:
                    os.remove(output_path)
                except OSError:
                    pass

        # Preparation -- 3. Load model for graph modification
        try:
            model = onnx.load(current_model_path)
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return None, None

        # Processing -- 2. Fix Dynamic Shape
        # Validation -- 2.1. Strict API Check for Dynamic Shapes (Ensure it is a dict)
        dyn_shape_cfg = json_strategies.get('fix_dynamic_shape', {})
        if dyn_shape_cfg and not isinstance(dyn_shape_cfg, dict):
            raise TypeError(
                f"YAML Error: 'fix_dynamic_shape' must be a dictionary, got {type(dyn_shape_cfg).__name__}.\n"
                "Please update your YAML config to include 'enabled' and 'strict_override' sub-keys."
            )

        # Processing -- 2.2. Strategy 2: Fix Dynamic Shape (Delegate to DynamicShapeFixer)
        if dyn_shape_cfg.get('enabled', False):
            logger.info("Strategy (2/4): Fixing Dynamic Shapes...")
            strict_override = dyn_shape_cfg.get('strict_override', False)

            shape_fixer = DynamicShapeFixer(model, model_shapes,
                                            strict_override)
            if shape_fixer.process():
                is_model_modified = True
        else:
            logger.info("Strategy (2/4): Dynamic Shape Fixing Disabled.")
        # -------------------------
        # if json_strategies.get('fix_dynamic_shape', False):
        #     logger.info("Strategy (2/4): Fixing Dynamic Shapes...")
        #     if self._fix_dynamic_shapes(model, model_shapes):
        #         is_model_modified = True
        # else:
        #     logger.info("Strategy (2/4): Dynamic Shape Fixing Disabled.")
        # -------------------------

        # Processing -- 3. Fix INT64 Type (Decoder specific)
        if json_strategies.get('fix_int64_type', False):
            logger.info("Strategy (3/4): Fixing INT64 Types for inputs...")
            if self._fix_int64_type(model):
                is_model_modified = True
        else:
            logger.info("Strategy (3/4): INT64 Type Fixing Disabled.")

        # Processing -- 4. Save intermediate if is_model_modified
        if is_model_modified:
            logger.debug(f"Saving intermediate fixed model to {output_path}")
            onnx.save(model, output_path)
            current_model_path = output_path

        # Processing -- 5. Simplify (onnxsim)
        if json_strategies.get('simplify', False):
            logger.info("Strategy (4/4): Running ONNX Simplifier...")
            current_model_path = self._simplify(current_model_path,
                                                output_path)
        else:
            logger.info("Strategy (4/4): ONNX Simplification Disabled.")

        # Finalization
        return current_model_path, custom_string
