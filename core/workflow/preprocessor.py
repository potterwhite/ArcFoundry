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
from core.quantization import strategies
from core.utils.utils import logger


class Preprocessor:
    """
    Handles all ONNX graph modifications before RKNN conversion.
    Strategies: Fix Dynamic Shape, Simplify, Extract Metadata, Fix Types.
    """

    def __init__(self, config):
        self.cfg = config

    # def _fix_dynamic_shapes(self, model, json_input_shapes):
    # """Iterates through inputs and sets dim_param to 1."""
    # changed = False
    # for input_tensor in model.graph.input:
    #     shape = input_tensor.type.tensor_type.shape
    #     if shape:
    #         for dim in shape.dim:
    #             if dim.dim_param:
    #                 logger.debug(f"  - Fixing dim '{dim.dim_param}' to 1 in input '{input_tensor.name}'")
    #                 dim.ClearField("dim_param")
    #                 dim.dim_value = 1
    #                 changed = True
    # return changed

    def _fix_dynamic_shapes(self, model, json_input_shapes, json_strict_override=False):
        """
        Fix dynamic input dimensions of an ONNX model
        using json_input_shapes provided from YAML configuration.

        Parameters
        ----------
        model : onnx.ModelProto
            Loaded ONNX model.

        json_input_shapes : list[list[int]]
            List of shapes defined in YAML.
            Each inner list corresponds to one model input.

            Example:
                [[1, 3, 512, 512]]
                [[1, 16000]]
                [[1, 80, 300]]

        Returns
        -------
        onnx.ModelProto
            Model with fixed input dimensions.
        """

        # ---------------------------------------------------------
        # 1️⃣ Basic validation of json_input_shapes structure
        # ---------------------------------------------------------
        if not isinstance(json_input_shapes, list):
            raise ValueError("json_input_shapes must be a list of shapes.")

        if len(json_input_shapes) == 0:
            raise ValueError("json_input_shapes cannot be empty.")

        # Ensure every entry is a list of integers
        for shape in json_input_shapes:
            if not isinstance(shape, list):
                raise ValueError("Each element in json_input_shapes must be a list.")

            for dim in shape:
                if not isinstance(dim, int):
                    raise ValueError("All dimensions in json_input_shapes must be integers.")

        # ---------------------------------------------------------
        # 2️⃣ Get ONNX model inputs
        # ---------------------------------------------------------
        model_inputs = model.graph.input

        if len(model_inputs) != len(json_input_shapes):
            raise ValueError(f"Input count mismatch: ONNX model has {len(model_inputs)} inputs, "
                             f"but YAML defines {len(json_input_shapes)} shapes.")

        # ---------------------------------------------------------
        # 3️⃣ Iterate over each input tensor
        # ---------------------------------------------------------
        for idx, input_tensor in enumerate(model_inputs):

            target_shape = json_input_shapes[idx]
            dims = input_tensor.type.tensor_type.shape.dim

            # Validate rank consistency
            if len(dims) != len(target_shape):
                raise ValueError(f"Rank mismatch for input '{input_tensor.name}': "
                                 f"Model expects {len(dims)} dims, "
                                 f"but YAML defines {len(target_shape)} dims.")

            # -----------------------------------------------------
            # 4️⃣ Replace dynamic dimensions safely
            # -----------------------------------------------------
            for i, dim in enumerate(dims):
                logger.debug(
                    f"  - Processing dimension index {i}: "
                    f"dim_param='{dim.dim_param}', dim_value={dim.dim_value}, "
                    f"target_value={target_shape[i]}"
                )

                # Replace dimension safely
                if json_strict_override:
                    # Force override all dimensions
                    dim.dim_param = ""
                    dim.dim_value = int(target_shape[i])
                else:
                    # Detect dynamic dimension:
                    # Case A: symbolic dimension (e.g., "batch", "height")
                    # Case B: undefined dimension (dim_value == 0)
                    is_dynamic = False

                    if dim.dim_param:
                        is_dynamic = True

                    if dim.dim_value == 0:
                        is_dynamic = True

                    # Only override dynamic dimensions
                    # This avoids accidentally changing fixed dimensions
                    if is_dynamic:
                        dim.dim_param = ""
                        dim.dim_value = int(target_shape[i])

            # Optional strict mode:
            # If you want to force override ALL dims, remove is_dynamic check.

        # # ---------------------------------------------------------
        # # 5️⃣ Re-run shape inference to stabilize graph
        # # ---------------------------------------------------------
        # model = onnx.shape_inference.infer_shapes(model)

        return model

    def _fix_int64_type(self, model):
        """Forces input type to INT64 (Crucial for Sherpa Decoder)."""
        changed = False
        for input_tensor in model.graph.input:
            # Heuristic: If it's the specific 'y' input or generally needed
            # For now, apply to 'y' as seen in original script, or make generic via config later.
            if input_tensor.name == 'y':
                logger.debug(f"  - Forcing input '{input_tensor.name}' to INT64")
                input_tensor.type.tensor_type.elem_type = onnx.TensorProto.INT64
                changed = True
        return changed

    def _simplify(self, input_path, output_path):
        """Wraps onnxsim."""
        try:
            model = onnx.load(input_path)
            model_simp, check = onnxsim.simplify(model)
            if check:
                onnx.save(model_simp, output_path)
                return output_path
            else:
                logger.warning("onnxsim check failed, using unsimplified model.")
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
            session = onnxruntime.InferenceSession(onnx_path,
                                                   sess_options,
                                                   providers=["CPUExecutionProvider"])

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

    def preprocess(self, model_name, model_path, model_shapes, output_path, json_strategies):
        """
        Applies a series of preprocessing strategies to the ONNX model.
        """
        # Preparation -- 1. Define Variables
        current_model_path = model_path
        custom_string = None
        modified = False

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
                logger.info("⏩ [FAST-FORWARD] Model is valid. Skipping complex preprocessing steps.\n")

                return output_path, custom_string
            except Exception:
                logger.warning(f"   Cached model corrupted or invalid ({e}). removing and regenerating...")
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
        if json_strategies.get('fix_dynamic_shape', False):
            logger.info("Strategy (2/4): Fixing Dynamic Shapes...")
            if self._fix_dynamic_shapes(model, model_shapes):
                modified = True
        else:
            logger.info("Strategy (2/4): Dynamic Shape Fixing Disabled.")

        # Processing -- 3. Fix INT64 Type (Decoder specific)
        if json_strategies.get('fix_int64_type', False):
            logger.info("Strategy (3/4): Fixing INT64 Types for inputs...")
            if self._fix_int64_type(model):
                modified = True
        else:
            logger.info("Strategy (3/4): INT64 Type Fixing Disabled.")

        # Processing -- 4. Save intermediate if modified
        if modified:
            logger.debug(f"Saving intermediate fixed model to {output_path}")
            onnx.save(model, output_path)
            current_model_path = output_path

        # Processing -- 5. Simplify (onnxsim)
        if json_strategies.get('simplify', False):
            logger.info("Strategy (4/4): Running ONNX Simplifier...")
            current_model_path = self._simplify(current_model_path, output_path)
        else:
            logger.info("Strategy (4/4): ONNX Simplification Disabled.")

        # Finalization
        return current_model_path, custom_string
