import onnx_graphsurgeon as gs
import onnx
import argparse
from pathlib import Path

def bypass_single_transpose_node(graph: gs.Graph, node_name: str) -> bool:
    """
    Finds a Transpose node by its name and bypasses it by reconnecting
    its input directly to its output's consumers.

    Args:
        graph (gs.Graph): The GraphSurgeon graph object to modify.
        node_name (str): The exact name of the Transpose node to bypass.

    Returns:
        bool: True if the node was found and bypassed, False otherwise.
    """
    target_node = None
    for node in graph.nodes:
        if node.name == node_name:
            target_node = node
            break

    if not target_node:
        print(f"[Error] Node named '{node_name}' not found in the graph.")
        return False

    if target_node.op != "Transpose":
        print(f"[Error] Node '{node_name}' found, but it is not a Transpose operator (OpType={target_node.op}). Aborting.")
        return False

    print(f"[INFO] Found target node: {target_node.name} (OpType: {target_node.op})")

    # --- Core Surgery: Bypassing the node ---
    # Get the single input tensor of the Transpose node.
    input_tensor = target_node.inputs[0]

    # Get the single output tensor of the Transpose node.
    output_tensor = target_node.outputs[0]

    # This is the key operation:
    # Find all nodes that consume the 'output_tensor' and make them
    # consume the 'input_tensor' instead.
    # We do this by iterating over all consumers of the output tensor and
    # replacing the input tensor in each of them.
    consumers = list(output_tensor.outputs)  # Create a copy for safe iteration
    for consumer in consumers:
        for i, tensor in enumerate(consumer.inputs):
            if tensor == output_tensor:
                consumer.inputs[i] = input_tensor

    # --- Preparation for Cleanup ---
    # The target_node is now isolated (its output is not used by any other node).
    # To allow graph.cleanup() to remove it, we must disconnect its outputs.
    target_node.outputs.clear()

    print(f"[INFO] Node '{node_name}' has been successfully bypassed.")
    return True


def main():

    """
    main function to parse cmdline arguments and run the optimization script.
    """

    parser = argparse.ArgumentParser(
        description = "a script to bypass a specified transpose node in an onnx model",
        formatter_class = argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--input_model", 
        type=Path, 
        required=True,
        help="Path to the input ONNX model file."
    )

    parser.add_argument(
        "--output_model", 
        type=Path, 
        required=True,
        help="Path to the output ONNX model file."
    )

    parser.add_argument(
        "--node_name", 
        type=str, 
        required=True,
        help="The name of the Transpose node to bypass (e.g., Transpose_29)."
    )

    args = parser.parse_args()

    if not args.input_model.is_file():
        print(f"[Error] Input model not found at: {args.input_model}")
        return

    print(f"[INFO] Loading model: {args.input_model}")
    graph = gs.import_onnx(onnx.load(str(args.input_model)))

    success = bypass_single_transpose_node(graph, args.node_name)

    if success:
        graph.cleanup()
        onnx.save(gs.export_onnx(graph), str(args.output_model))
        print(f"[SUCCESS] modified model saved to: {args.output_model}")
        print("please verify the new model with netron and check its numerical accuracy.")
    else:
        print("[Failed] no modifications were made to the model.")

if __name__ == "__main__":
    main()



