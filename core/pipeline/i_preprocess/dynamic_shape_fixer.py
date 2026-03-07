from utils import logger


class DynamicShapeFixer:
    """
    Handles resolving dynamic dimensions in ONNX models using a data-driven approach.
    Single Responsibility: Modify tensor shape definitions in the ONNX Graph.
    """

    def __init__(self, model, json_input_shapes, strict_override):
        self.model = model
        self.json_input_shapes = json_input_shapes
        self.strict_override = strict_override
        self.model_modified = False

    def process(self):
        """
        Main controller for shape modification.
        Routes each tensor to either the explicit or fallback strategy.

        Returns:
            bool: True if the model was modified, False otherwise.
        """
        # Validation -- 1. Enforce Strict API (Must be a dictionary)
        if not isinstance(self.json_input_shapes, dict):
            logger.error(
                f"json_input_shapes={self.json_input_shapes} is not a dictionary."
            )
            raise ValueError("json_input_shapes MUST be a dictionary.\n"
                             "Update your YAML format. Example:\n"
                             "input_shapes:\n"
                             "  input_tensor_name: [1, 3, 224, 224]")

        # Processing -- 2. Iterate Tensors (Loop through all ONNX graph inputs)
        for tensor_idx, input_tensor in enumerate(self.model.graph.input,
                                                  start=1):
            tensor_name = input_tensor.name

            # Routing -- 3. Path A: Explicit Mapping (Tensor name exists in YAML)
            if tensor_name in self.json_input_shapes:
                target_shape = self.json_input_shapes[tensor_name]
                if self._apply_explicit_shape(input_tensor, tensor_name,
                                              target_shape, tensor_idx):
                    self.model_modified = True

            # Routing -- 4. Path B: Implicit Fallback (Tensor name NOT in YAML)
            else:
                if self._apply_fallback_shape(input_tensor, tensor_name,
                                              tensor_idx):
                    self.model_modified = True

        # Validation -- 5. Final Status Check (Log success if changes occurred)
        if self.model_modified:
            logger.info("\tSuccessfully fixed dynamic shapes for model inputs.")

        return self.model_modified

    def _apply_explicit_shape(self, tensor, name, target_shape, idx):
        """
        Specialized Method: Overrides dimensions based on explicit YAML definitions.
        """
        dims = tensor.type.tensor_type.shape.dim
        modified = False

        # Validation -- 6. Rank Check (Ensure ONNX and YAML dimensions match in length)
        if len(dims) != len(target_shape):
            raise ValueError(
                f"Rank mismatch for '{name}': "
                f"ONNX expects {len(dims)} dims, YAML provides {len(target_shape)} dims."
            )

        # Processing -- 7. Apply Target Dimensions (Override if dynamic or strictly forced)
        for i, dim in enumerate(dims):
            is_dynamic = bool(dim.dim_param) or (dim.dim_value == 0)

            if self.strict_override or is_dynamic:
                logger.info(
                    f"  - [Explicit] Fixing [{idx}]-'{name}' dim {i} to {target_shape[i]}"
                )
                dim.dim_param = ""
                dim.dim_value = int(target_shape[i])
                modified = True

        return modified

    def _apply_fallback_shape(self, tensor, name, idx):
        """
        Specialized Method: Defaults any unmapped dynamic dimensions to 1.
        """
        dims = tensor.type.tensor_type.shape.dim
        modified = False

        # Processing -- 8. Apply Fallback (Set unmapped symbolic dims to 1)
        for i, dim in enumerate(dims):
            is_dynamic = bool(dim.dim_param) or (dim.dim_value == 0)

            if is_dynamic:
                logger.debug(
                    f"  - [Fallback] Auto-fixing unmapped dynamic dim in [{idx}]-'{name}' to 1"
                )
                dim.dim_param = ""
                dim.dim_value = 1
                modified = True

        return modified
