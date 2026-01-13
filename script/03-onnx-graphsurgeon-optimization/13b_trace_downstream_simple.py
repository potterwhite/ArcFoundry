import onnx_graphsurgeon as gs
import onnx
from collections import deque
from pathlib import Path

ONNX_FILE = Path("/development/asr/build-scripts/onnx/c-convert/models/onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx")
#START_NODE_NAME = "Transpose_41"
#START_NODE_NAME = "Transpose_199"
#START_NODE_NAME = "Transpose_285"
#START_NODE_NAME = "Transpose_302"
#START_NODE_NAME = "Transpose_201"
#START_NODE_NAME = "Transpose_211"
START_NODE_NAME = "Transpose_29"

MAX_DEPTH = 15

if not ONNX_FILE.is_file():
    print(f"[Error] model file not found at {ONNX_FILE}")
else:
    try:
        graph = gs.import_onnx(onnx.load(str(ONNX_FILE)))

        start_node = None
        for node in graph.nodes:
            if node.name == START_NODE_NAME:
                start_node = node
                break

        if not start_node:
            print(f"[Error]start node '{START_NODE_NAME}' not found in the graph.")
        else:
            print(f"--- tracing from {start_node.name} (OpType: {start_node.op}) ---\n")

            #breadth first search
            queue = deque([(start_node, 0)])
            visited_nodes = {start_node.name}

            while queue:
                current_node, current_depth = queue.popleft()

                if current_depth >= MAX_DEPTH:
                    print(f"{'  ' * (current_depth+1)}-> [END of PATH] (Tensor: {output_tensor.name})")
                    continue

                for output_tensor in current_node.outputs:
                    if not output_tensor.outputs:
                        print(f"{'  ' * (current_depth + 1)}-> [END OF PATH] (Tensor: {output_tensor.name})")
                    else:
                        for consumer_node in output_tensor.outputs:
                            indent = "  " * (current_depth + 1)

                            perm_info = ""
                            if consumer_node.op == "Transpose" and 'perm' in consumer_node.attrs:
                                perm_info = f" (perm={consumer_node.attrs['perm']})"

                            print(f"{indent}-> node: {consumer_node.name} (optype: {consumer_node.op}){perm_info}")

                            if consumer_node.name not in visited_nodes:
                                visited_nodes.add(consumer_node.name)
                                queue.append((consumer_node, current_depth + 1))                

    except Exception as e:
        print(f"[Error] An error occured: {e}")
