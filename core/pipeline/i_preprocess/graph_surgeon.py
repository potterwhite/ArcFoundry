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

import onnx_graphsurgeon as gs
import numpy as np
from utils import logger


class GraphSurgeon:
    """
    Modular graph surgery module powered by onnx-graphsurgeon.

    Sub-operations organized by OGS's four manipulation targets:
      - outputs: graph.outputs (implemented)
      - inputs:  graph.inputs  (future)
      - nodes:   graph.nodes   (future)
      - tensors: tensor attrs  (future)

    Each sub-object has operation verbs (modify / add / remove),
    each verb takes a list of parameter dicts.

    Config format:
        graph_surgery:
          enabled: true
          outputs:
            modify:
              - existing: "fgr"        # tensor name currently in graph.outputs
                replacement: "777"     # tensor name to replace it with
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def process(self):
        """
        Execute all configured graph surgery sub-operations.

        Returns:
            bool: True if the model was modified, False otherwise.
        """
        graph = gs.import_onnx(self.model)
        modified = False

        # Sub-operation: outputs.modify — replace tensors in graph.outputs
        if 'outputs' in self.config:
            outputs_cfg = self.config['outputs']
            if 'modify' in outputs_cfg:
                modified |= self._modify_outputs(graph, outputs_cfg['modify'])

        if modified:
            graph.cleanup().toposort()
            self.model = gs.export_onnx(graph)

        return modified

    def _modify_outputs(self, graph, replacements):
        """
        Replace tensors in graph.outputs list.

        For each replacement rule:
          1. Find the tensor named `existing` in graph.outputs
          2. Replace it with the tensor named `replacement` from the graph

        After replacement, graph.cleanup() (called by process()) automatically
        removes all dead nodes that no longer contribute to any output.

        Args:
            graph: onnx_graphsurgeon.Graph
            replacements: list of {existing: str, replacement: str}

        Returns:
            bool: True if any output was replaced
        """
        # Build tensor name → tensor object index (from all node outputs)
        tensor_map = {}
        for node in graph.nodes:
            for out in node.outputs:
                tensor_map[out.name] = out

        # Also index graph inputs (replacement target might be an input)
        for inp in graph.inputs:
            tensor_map[inp.name] = inp

        replaced_any = False
        for rule in replacements:
            old_name = rule.get('existing', '')
            new_name = rule.get('replacement', '')

            if not old_name or not new_name:
                logger.error(f"graph_surgery: invalid rule {rule}, need 'existing' and 'replacement'")
                return False

            if new_name not in tensor_map:
                logger.error(f"graph_surgery: tensor '{new_name}' not found in graph")
                return False

            # Find and replace in graph.outputs
            found = False
            for i, out in enumerate(graph.outputs):
                if out.name == old_name:
                    graph.outputs[i] = tensor_map[new_name]
                    logger.info(f"  - Replaced output '{old_name}' -> '{new_name}'")
                    found = True
                    replaced_any = True
                    break

            if not found:
                logger.warning(f"  - Output '{old_name}' not found in graph.outputs, skipping")

        return replaced_any

    def get_model(self):
        return self.model
