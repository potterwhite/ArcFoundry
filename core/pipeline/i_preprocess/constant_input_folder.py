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

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
from utils import logger


class ConstantInputFolder:
    """
    Promotes selected graph inputs into initializer constants.

    Problem this solves
    -------------------
    Some models (e.g. RVM) expose scalar control inputs (e.g. ``downsample_ratio``)
    as graph inputs rather than baking them in as initializers.  RKNN Toolkit 2's
    ``fold_constant`` pass requires every non-data path to be traceable to a
    constant; a live graph input blocks that propagation and raises:

        ValueError: The input 2 of Resize('Resize_3') need to be constant!

    Strategy
    --------
    For each tensor name listed in ``config.inputs``:
      1. Remove it from ``model.graph.input``  (stops being a live input)
      2. Add a matching initializer with the supplied value  (becomes a constant)

    After this pass, onnxsim's constant-folding can collapse downstream nodes
    (Concat → Resize scales, Shape → Slice → Concat → Resize sizes, etc.) into
    plain constants that RKNN accepts.

    Usage (YAML)
    ------------
    preprocess:
      fold_constant_inputs:
        enabled: true
        inputs:
          downsample_ratio: [0.25]   # list of float values

    Notes
    -----
    - Only scalar / 1-D float tensors are supported (sufficient for all known
      use-cases; extend dtype mapping if needed).
    - If a name in ``inputs`` does not exist in ``graph.input``, a warning is
      emitted and processing continues — prevents silent breakage when the
      model is later updated and the input disappears.
    """

    # Mapping from Python list element type → ONNX TensorProto dtype
    _DTYPE_MAP = {
        float: TensorProto.FLOAT,
        int:   TensorProto.INT64,
    }

    def __init__(self, model, config_inputs: dict):
        """
        Args:
            model:          Loaded ``onnx.ModelProto``.
            config_inputs:  Dict mapping tensor-name → list-of-values, e.g.
                            ``{"downsample_ratio": [0.25]}``.
        """
        self.model = model
        self.config_inputs = config_inputs
        self.model_modified = False

    def process(self) -> bool:
        """
        Execute the folding pass.

        Returns:
            True if at least one input was promoted to a constant.
        """
        if not self.config_inputs:
            logger.debug("ConstantInputFolder: no inputs configured, skipping.")
            return False

        # Build a fast lookup of current graph input names
        graph_input_names = {t.name for t in self.model.graph.input}

        for tensor_name, values in self.config_inputs.items():

            # Validation -- 1. Name must exist in graph inputs
            if tensor_name not in graph_input_names:
                logger.warning(
                    f"  - [ConstantInputFolder] '{tensor_name}' not found in "
                    f"graph.input — skipping (model may have changed)."
                )
                continue

            # Validation -- 2. Values must be a non-empty list
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(
                    f"fold_constant_inputs.inputs['{tensor_name}'] must be a "
                    f"non-empty list, got: {values!r}"
                )

            # Processing -- 3. Determine dtype from first element
            first = values[0]
            dtype = self._DTYPE_MAP.get(type(first))
            if dtype is None:
                raise TypeError(
                    f"fold_constant_inputs.inputs['{tensor_name}']: "
                    f"unsupported element type {type(first).__name__!r}. "
                    f"Supported: float, int."
                )

            # Processing -- 4. Build numpy array and wrap as initializer
            if dtype == TensorProto.FLOAT:
                np_array = np.array(values, dtype=np.float32)
            else:
                np_array = np.array(values, dtype=np.int64)

            initializer = numpy_helper.from_array(np_array, name=tensor_name)
            self.model.graph.initializer.append(initializer)

            # Processing -- 5. Remove from graph.input
            inputs_to_keep = [
                t for t in self.model.graph.input if t.name != tensor_name
            ]
            del self.model.graph.input[:]
            self.model.graph.input.extend(inputs_to_keep)

            logger.info(
                f"  - [ConstantInputFolder] Promoted '{tensor_name}' "
                f"→ constant {np_array.tolist()}"
            )
            self.model_modified = True

        return self.model_modified
