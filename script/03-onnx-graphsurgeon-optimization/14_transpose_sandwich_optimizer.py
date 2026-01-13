import onnx
import onnx_graphsurgeon as gs
import numpy as np
from collections import deque
from pathlib import Path

INPUT_MODEL_PATH = Path("/development/asr/build-scripts/onnx/c-convert/models/onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx")
OUTPUT_MODEL_PATH = Path("/development/asr/build-scripts/onnx/c-convert/models/onnx_modified/transpose_inverse.onnx")

def func_4_0_get_inverse_permutation(perm):
    inverse_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse_perm[p] = i
    return tuple(inverse_perm)

def func_4_1_remap_axis(axis, perm):
    if axis < 0:
        axis += len(perm)
    return perm.index(axis)

def func_4_2_transform_conv_weights(node, perm):
    print(f"    func_4_2: transform ConvOp Weights, nodename={node.name} ({node.op}) perm={perm}")

    if len(node.inputs) < 2:
        print(f"   weight does not exist, exit")
        return False

    if not isinstance(node.inputs[1], gs.Constant):
        print(f"    weight is not gs.Constant type,exit")
        return False

    weights = node.inputs[1]
    w_shape = weights.values.shape

    weight_perm = None
    if len(w_shape) == 3:
        if perm == (1, 2, 0) or perm == (2, 0, 1):
            weight_perm = (0, 2, 1)

    if weight_perm:
        print(f"      Applying weight perm {weight_perm} to shape {w_shape}")
        weights.values = np.transpose(weights.values, weight_perm)
        print(f"      New weight shape: {weights.values.shape}")
        return True
    else:
        print(f"      [Fail] No transformation logic for data perm {perm} and weight shape {w_shape}.")
        return False

    



def func_3_0_analyze_and_transform_subgraph(subgraph_nodes, perm):
    print(f"    [->] func_3_0: Analyzing subgraph of {len(subgraph_nodes)} nodes for perm={perm}...")

#    LAO_PARAMETRIC = {"Split", "Concat", "ReduceMean", "ReduceSum", "Slice", "Gather", "Unsqueeze", "Squeeze"}
#    LAO_PURE = {"Add", "Sub", "Mul", "Div", "Sigmoid", "ReLU", "Softmax", "Cast", 
    # add EWO Shape for current model
#    "Shape", 
    # add more EWO for the future
#    "ConstantOfShape", "Range", "Tile", "Expand", "Pow",
#    }
#    LSO_GENERIC = {"Reshape", "Transpose", "MatMul", "Conv"}
#    LSO_SPECIALIZED = {}
    # ===================================================================================
    # == OPERATOR CLASSIFICATION FRAMEWORK
    # ===================================================================================
    # This framework categorizes ONNX operators based on their sensitivity to the
    # tensor data layout (e.g., NCHW vs. NHWC). This classification is the core
    # logic that determines whether a 'Transpose sandwich' can be safely optimized.
    #
    # Abbreviations:
    #   - LAO: Layout-Agnostic Operator (布局无关算子)
    #   - LSO: Layout-Sensitive Operator (布局敏感算子)
    # ===================================================================================

    # -----------------------------------------------------------------------------------
    # 1. LAO_PURE (Pure Layout-Agnostic Operators)
    # -----------------------------------------------------------------------------------
    # Definition: Operators whose computation is entirely independent of the tensor's
    # memory layout. They are typically Element-Wise Operators (EWO).
    #
    # Key Characteristic: Their operation is point-to-point and requires no information
    # about neighboring elements or dimensional arrangement.
    #
    # Handling Strategy: SAFE TO PASS. These operators require no modification. The script
    # will simply 'continue' when it encounters them.
    # -----------------------------------------------------------------------------------
    LAO_PURE = {
        "Add", "Sub", "Mul", "Div",        # Basic Arithmetic EWO
        "Pow", "Sqrt", "Exp", "Log",       # Mathematical EWO
        "Sigmoid", "ReLU", "Tanh", "Cast", # Activation & Type Casting EWO
        "Shape",                           # Returns shape info, agnostic to data layout itself
        "ConstantOfShape", "Range", "Tile", "Expand" # Tensor generation, behavior is layout-agnostic
    }

    # -----------------------------------------------------------------------------------
    # 2. LAO_PARAMETRIC (Parametric Layout-Agnostic Operators)
    # -----------------------------------------------------------------------------------
    # Definition: Operators that can behave as LAOs on the condition that their
    # axis-dependent parameters (e.g., 'axis', 'axes') are correctly transformed
    # to match the new data layout.
    #
    # Key Characteristic: Their layout dependency is fully encapsulated within their
    # attributes.
    #
    # Handling Strategy: SAFE TO TRANSFORM. The script will remap their 'axis' or 'axes'
    # attributes before allowing the optimization to proceed.
    # -----------------------------------------------------------------------------------
    LAO_PARAMETRIC = {
        "Split", "Concat",                 # Dimension-wise joining/splitting
        "ReduceMean", "ReduceSum",         # Dimension reduction
        "Slice", "Gather",                 # Data slicing and gathering
        "Unsqueeze", "Squeeze",            # Dimension manipulation
        "Softmax"                          # Normalization along a specific axis
    }

    # -----------------------------------------------------------------------------------
    # 3. LSO_GENERIC (Generic Layout-Sensitive Operators)
    # -----------------------------------------------------------------------------------
    # Definition: Operators whose internal logic is strongly coupled with the data layout
    # in a way that cannot be easily or generically compensated for by simple attribute changes.
    #
    # Key Characteristic: They make strong assumptions about the order and contiguity
    # of data in memory (e.g., matrix multiplication, spatial sliding windows).
    #
    # Handling Strategy: BLOCKING. If any of these operators are found in the subgraph,
    # the optimization for that specific 'Transpose sandwich' is aborted (returns False).
    # -----------------------------------------------------------------------------------
    LSO_GENERIC = {
        "Reshape", "Transpose", "Flatten", # Explicit layout transformation
        "MatMul",                           # Matrix multiplication
        "Conv"
    }

    # -----------------------------------------------------------------------------------
    # 4. LSO_SPECIALIZED (Specialized Layout-Sensitive Operators)
    # -----------------------------------------------------------------------------------
    # Definition: A subset of LSOs that are fundamentally layout-sensitive but might be
    # handled through complex, specialized transformations (e.g., weight rewriting).
    #
    # Key Characteristic: These represent advanced optimization targets. Our current script
    # classifies them as blocking, but they are kept separate to signify potential for
    # future, more sophisticated optimization strategies.
    #
    # Handling Strategy: BLOCKING (for now). Currently treated the same as LSO_GENERIC.
    # The 'Conv' operator failed our previous weight transformation attempt, proving its
    # specialized nature.
    # -----------------------------------------------------------------------------------
    LSO_SPECIALIZED = {
#        "Conv"
    }

    for node in subgraph_nodes:
        if node.op in LSO_SPECIALIZED:
            if not func_4_2_transform_conv_weights(node, perm):
                return False
            continue

        if node.op in LAO_PURE:
            print(f"   [Process] Found an agnostic operator '{node.op}' in node '{node.name}'")
            continue

        if node.op in LAO_PARAMETRIC:
            if 'axis' in node.attrs:
                old_axis = node.attrs['axis']
                new_axis = func_4_1_remap_axis(old_axis, perm)
                node.attrs['axis'] = new_axis
                print(f"    - Remapped axis for {node.name} ({node.op}): {old_axis} -> {new_axis}")
                continue
            elif 'axes' in node.attrs:
                old_axes = node.attrs['axes']
                new_axes = [func_4_1_remap_axis(ax, perm) for ax in old_axes]
                node.attrs['axes'] = new_axes
                print(f"    - Remapped axes for {node.name} ({node.op}): {old_axis} -> {new_axis}")
                continue
            else:
                print("     - OMG...not axis not axes, neither")
                continue

        if node.op in LSO_GENERIC:
            print(f"   [FAIL] Aborting: Found an blocking operator '{node.op}' in node '{node.name}'")
            return False

        print(f"   [FAIL] Aborting: Found an unhandled operator '{node.op}' in node '{node.name}'")
        return False
    
    print("   [OK] Subgraph analysis successful. Transformation is safe")
    return True


def func_2_0_find_transpose_sandwiches(graph: gs.Graph):
    print("  [->] func_2_0: Starting to find all Transpose sandwiches...")
    sandwiches = []

    for node in graph.nodes:
        if node.op != "Transpose" or not node.attrs.get("perm"):
            continue

        entry_node = node
        entry_perm = tuple(entry_node.attrs["perm"])
        inverse_perm = func_4_0_get_inverse_permutation(entry_perm)

        subgraph_nodes = []
        exit_node = None

        queue = deque(entry_node.outputs[0].outputs)
        visited_nodes = {entry_node.name}

        while queue:
            current_node = queue.popleft()
            if current_node.name in visited_nodes:
                continue
            visited_nodes.add(current_node.name)

            if current_node.op == "Transpose" and tuple(current_node.attrs.get("perm", [])) == inverse_perm:
                is_clean_exit = all(inp.inputs[0].name in visited_nodes for inp in current_node.inputs if inp.inputs)
                if is_clean_exit:
                    exit_node = current_node
                    break

            subgraph_nodes.append(current_node)
            for out_tensor in current_node.outputs:
                for consumer_node in out_tensor.outputs:
                    if consumer_node.name not in visited_nodes:
                        queue.append(consumer_node)

        if exit_node:
            print(f"    - Found a sandwich: {entry_node.name} -> {exit_node.name}")
            sandwiches.append((entry_node, exit_node, subgraph_nodes))

    print(f" [<-] func_2_0: Found {len(sandwiches)} sandwiches in total.")
    return sandwiches

def func_1_0_main_optimizer(graph: gs.Graph):
    print("  [->] func_1_0: starting main optimizer")

    sandwiches = func_2_0_find_transpose_sandwiches(graph)

    if not sandwiches:
        print("[<-] func_1_0: no optimizable patterns found. exiting.")
        return graph

    node_modified = False
    for entry_node, exit_node, subgraph_nodes in sandwiches:
        print(f"\n  Processing sandwich: {entry_node.name} <-> {exit_node.name}")

        is_safe_to_transform = func_3_0_analyze_and_transform_subgraph(subgraph_nodes, tuple(entry_node.attrs["perm"]))

        if is_safe_to_transform:
            #---------------
            # obsoleted
            #entry_node.outputs[0].reconnect(entry_node.inputs[0])
            #exit_node.outputs[0].reconnect(exit_node.inputs[0])
            #-----------
            inp_tensor_entry = entry_node.inputs[0]
            out_tensor_entry = entry_node.outputs[0]

            for consumer in list(out_tensor_entry.outputs):
                for i, t in enumerate(consumer.inputs):
                    if t == out_tensor_entry:
                        consumer.inputs[i] = inp_tensor_entry
            entry_node.outputs.clear()


            inp_tensor_exit = exit_node.inputs[0]
            out_tensor_exit = exit_node.outputs[0]
            for consumer in list(out_tensor_exit.outputs):
                for i, t in enumerate(consumer.inputs):
                    if t == out_tensor_exit:
                        consumer.inputs[i] = inp_tensor_exit
            exit_node.outputs.clear()


            print(f"  [SUCCESS] Successfully rewired graph to bypass {entry_node.name} and {exit_node.name}.")
            node_modified = True

    if node_modified:
        print("\n  Cleaning up graph to remove bypassed nodes...")
        graph.cleanup()
        print("  Graph cleanup complete.")

    print("[<-] func_1_0: Optimizer finished.")
    return graph


if __name__ == "__main__":
    if not INPUT_MODEL_PATH.is_file():
        print(f"[FATAL] Input model not found at: {INPUT_MODEL_PATH}")
    else:
        print(f"Loading ONNX model: {INPUT_MODEL_PATH}")
        graph = gs.import_onnx(onnx.load(str(INPUT_MODEL_PATH)))

        optimized_graph = func_1_0_main_optimizer(graph)

        output_model_dir = OUTPUT_MODEL_PATH.parent
        output_model_dir.mkdir(parents=True, exist_ok=True)
        onnx.save(gs.export_onnx(optimized_graph), str(OUTPUT_MODEL_PATH))
        print(f"\n Optimization complete. New model saved to: {OUTPUT_MODEL_PATH}")

















