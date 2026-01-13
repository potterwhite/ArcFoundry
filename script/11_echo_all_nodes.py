# FILE: inspect_onnx_models.py

import onnx_graphsurgeon as gs
import onnx
import argparse
from pathlib import Path

def inspect_single_model(model_path: Path, verbose: bool):
    """
    Loads and inspects a single ONNX model file.

    Args:
        model_path (Path): The path to the ONNX model file.
        verbose (bool): If True, prints detailed information for every node.
                        Otherwise, prints only a summary.
    """
    if not model_path.is_file():
        print(f"--- [SKIPPING] File not found: {model_path} ---\n")
        return

    print(f"--- [INSPECTING] Model: {model_path.name} ---")
    
    try:
        # Load the ONNX model into a GraphSurgeon Graph object.
        graph = gs.import_onnx(onnx.load(str(model_path)))

        print("  --- Model Inputs ---")
        for inp in graph.inputs:
            print(f"    Name: {inp.name}, DType: {inp.dtype}, Shape: {inp.shape}")

        print("\n  --- Model Outputs ---")
        for out in graph.outputs:
            print(f"    Name: {out.name}, DType: {out.dtype}, Shape: {out.shape}")
        
        print(f"\n  Summary: Model has {len(graph.nodes)} nodes (operators).")

        if verbose:
            print("\n  --- All Nodes (Operators) in Detail ---")
            # Iterate through all nodes to print their detailed structure.
            for i, node in enumerate(graph.nodes):
                print(f"    --- Node {i} ---")
                print(f"      Name: {node.name}")
                print(f"      OpType: {node.op}")
                print(f"      Inputs: {[i.name for i in node.inputs]}")
                print(f"      Outputs: {[o.name for o in node.outputs]}")
                # For even more detail, you can inspect node.attrs
                # print(f"      Attributes: {node.attrs}")
        
        print(f"--- [DONE] Finished inspecting {model_path.name} ---\n")

    except Exception as e:
        print(f"[ERROR] Failed to inspect model {model_path.name}: {e}\n")


def main():
    """
    Main function to parse command-line arguments and run the inspection.
    """
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Inspect one or more ONNX models using ONNX-GraphSurgeon.",
        formatter_class=argparse.RawTextHelpFormatter # For better help message formatting
    )

    # Add the arguments
    parser.add_argument(
        "model_paths",
        metavar="MODEL_PATH",
        type=Path,
        nargs="+",  # This allows one or more arguments
        help="Path(s) to the ONNX model file(s) to inspect."
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",  # This makes it a flag, e.g., -v
        help="Print detailed information for every node in the graph."
    )

    # Execute the parse_args() method
    args = parser.parse_args()

    # Loop through the provided model paths and inspect each one
    for model_path in args.model_paths:
        inspect_single_model(model_path, args.verbose)


if __name__ == "__main__":
    main()
