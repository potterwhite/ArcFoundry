import onnx_graphsurgeon as gs
import onnx

graph = gs.import_onnx(onnx.load("/development/asr/build-scripts/onnx/c-convert/models/onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx"))

transpose_patterns = {}

for node in graph.nodes:
    if node.op == "Transpose":
        perm_tuple = tuple(node.attrs['perm'])

        if perm_tuple not in transpose_patterns:
            transpose_patterns[perm_tuple] = []

        transpose_patterns[perm_tuple].append(node.name)


print("--- Transpose pattern analysis ---")
for perm, nodes in transpose_patterns.items():
    print(f"Pattern perm={perm}: found {len(nodes)} times.")
    print(f"    Nodes: {nodes[:1]}")
