#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
#      FILE: inspect_quant_config.py
#
#     USAGE: See `func_3_0_setup_arg_parser()` or run with --help
#
#   DESCRIPTION: An advanced utility to inspect RKNN quantization configs (.cfg)
#                and cross-reference them with an ONNX model. It can provide
#                summaries, filter layers, and automatically find layers in the
#                .cfg file corresponding to specific operator types (e.g., Softmax)
#                from the ONNX graph.
#
#       NOTES: Requires `onnx` library. Install with: pip install onnx
#      AUTHOR: (Your Name or Alias)
#     LICENSE: MIT License
#     VERSION: 2.0.0
#     CREATED: 2023-09-22
#    REVISION: 2023-09-23
# ==============================================================================

import argparse
import sys
import re
from collections import defaultdict

try:
    import onnx
except ImportError:
    print(
        "Error: The 'onnx' library is required. Please install it using 'pip install onnx'",
        file=sys.stderr,
    )
    sys.exit(1)

# ==============================================================================
# Level 0: Constants and Global Configuration
# ==============================================================================

# --- Configuration: List of keywords for the --export-sensitive basic search ---
SENSITIVE_LAYER_KEYWORDS = [
    "softmax",
    "layernorm",
    "norm",
    "attention",
    "div",
    "exp",
    "log",
    "gelu",
]


class AnsiColors:
    """Helper class for terminal colors."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


# ==============================================================================
# Level 1: Core Utility and Parsing Functions
# ==============================================================================


def func_1_0_print_colored(text, color, file=sys.stdout):
    """
    Prints text to the specified file stream in a given color.

    Args:
        text (str): The text to print.
        color (str): ANSI color code from the AnsiColors class.
        file (stream): The output stream (default: sys.stdout).
    """
    print(f"{color}{text}{AnsiColors.ENDC}", file=file)


def func_1_1_parse_cfg_file(filepath):
    """
    Parses the quantization config file into a dictionary of layers.

    Args:
        filepath (str): Path to the .quantization.cfg file.

    Returns:
        dict: A dictionary mapping layer names to their properties.
              Returns None on failure.
    """
    layers = {}
    current_layer_name = None
    try:
        with open(filepath, "r") as f:
            found_start = False
            for line in f:
                if line.strip() == "quantize_parameters:":
                    found_start = True
                    break

            if not found_start:
                func_1_0_print_colored(
                    "[ERROR] Could not find 'quantize_parameters:' section in the config file.",
                    AnsiColors.FAIL,
                    file=sys.stderr,
                )
                return None

            for line in f:
                line_stripped = line.rstrip()
                if not line_stripped:
                    continue

                if line.startswith("    ") and not line.startswith("        "):
                    current_layer_name = line_stripped.strip().replace(":", "")
                    layers[current_layer_name] = {}
                elif current_layer_name and line.startswith("        ") and ":" in line:
                    parts = [p.strip() for p in line_stripped.split(":", 1)]
                    if len(parts) == 2:
                        key, value = parts
                        if key == "dtype":
                            layers[current_layer_name][key] = value
    except FileNotFoundError:
        func_1_0_print_colored(
            f"[ERROR] File not found: {filepath}", AnsiColors.FAIL, file=sys.stderr
        )
        return None
    except Exception as e:
        func_1_0_print_colored(
            f"[ERROR] Failed to parse config file: {e}",
            AnsiColors.FAIL,
            file=sys.stderr,
        )
        return None

    return layers


def func_1_2_load_onnx_model(filepath):
    """
    Loads an ONNX model from the specified file path.

    Args:
        filepath (str): Path to the .onnx model file.

    Returns:
        onnx.ModelProto: The loaded ONNX model object.
                         Returns None on failure.
    """
    try:
        model = onnx.load(filepath)
        return model
    except FileNotFoundError:
        func_1_0_print_colored(
            f"[ERROR] ONNX model not found: {filepath}",
            AnsiColors.FAIL,
            file=sys.stderr,
        )
        return None
    except Exception as e:
        func_1_0_print_colored(
            f"[ERROR] Failed to load ONNX model: {e}", AnsiColors.FAIL, file=sys.stderr
        )
        return None


def func_1_3_get_op_type_from_name(layer_name):
    """
    Heuristically guesses the operator type from its name in the CFG file.

    Args:
        layer_name (str): The name of the layer from the config file.

    Returns:
        str: The guessed operator type.
    """
    name_lower = layer_name.lower()

    priority_keywords = {
        "softmax": "Softmax",
        "layernorm": "LayerNorm",
        "attention": "Attention",
        "matmul": "MatMul",
        "matmult": "MatMul",
        "conv": "Conv",
        "relu": "ReLU",
        "sigmoid": "Sigmoid",
        "add": "Add",
        "mul": "Mul",
        "div": "Div",
        "gelu": "GELU",
        "norm": "LayerNorm",
    }

    for keyword, op_type in priority_keywords.items():
        if keyword in name_lower:
            return op_type

    parts = re.split(r"[._]", layer_name)
    meaningful_parts = [p for p in parts if not p.isdigit() and len(p) > 2]
    if meaningful_parts:
        return meaningful_parts[-1].capitalize()

    return "Unknown"


# ==============================================================================
# Level 2: High-Level Action Functions (Modes of Operation)
# ==============================================================================


def func_2_0_run_summary(cfg_data):
    """
    Analyzes CFG data and prints a summary of operator types and their dtypes.

    Args:
        cfg_data (dict): Parsed data from the config file.
    """
    summary = defaultdict(lambda: defaultdict(int))
    for name, properties in cfg_data.items():
        op_type = func_1_3_get_op_type_from_name(name)
        dtype = properties.get("dtype", "N/A")
        summary[op_type][dtype] += 1

    func_1_0_print_colored("\n--- Operator Summary ---", AnsiColors.HEADER)
    print(
        f"{'Operator Type':<25} | {'INT8 Count':<15} | {'FP32 Count':<15} | {'Other Count':<15}"
    )
    print("-" * 75)

    for op_type in sorted(summary.keys()):
        counts = summary[op_type]
        int8_count = counts.get("int8", 0)
        fp32_count = counts.get("float32", 0)
        other_count = sum(v for k, v in counts.items() if k not in ["int8", "float32"])

        int8_str = f"{AnsiColors.OKGREEN}{int8_count}{AnsiColors.ENDC}"
        fp32_str = f"{AnsiColors.FAIL}{fp32_count}{AnsiColors.ENDC}"
        other_str = f"{AnsiColors.WARNING}{other_count}{AnsiColors.ENDC}"

        print(f"{op_type:<25} | {int8_str:<25} | {fp32_str:<25} | {other_str:<25}")
    print("-" * 75)


def func_2_1_run_filter(cfg_data, filters):
    """
    Prints details for layers whose names contain any of the filter keywords.

    Args:
        cfg_data (dict): Parsed data from the config file.
        filters (list): A list of keywords to search for.
    """
    func_1_0_print_colored(
        f"\n--- Filtering for layers containing: {', '.join(filters)} ---",
        AnsiColors.HEADER,
    )
    found_count = 0
    for name, properties in cfg_data.items():
        if any(f.lower() in name.lower() for f in filters):
            dtype = properties.get("dtype", "N/A")
            color = (
                AnsiColors.OKGREEN
                if dtype == "int8"
                else AnsiColors.FAIL if dtype == "float32" else AnsiColors.WARNING
            )
            print(f"Layer: {name:<80} DType: ", end="")
            func_1_0_print_colored(dtype, color)
            found_count += 1

    if found_count == 0:
        func_1_0_print_colored("No matching layers found.", AnsiColors.WARNING)
    else:
        print(f"\nFound {found_count} matching layers.")


def func_2_2_run_export_sensitive(cfg_data):
    """
    Finds and prints layers matching a predefined list of sensitive keywords.

    Args:
        cfg_data (dict): Parsed data from the config file.
    """
    sensitive_layers = []
    for name in cfg_data.keys():
        if any(keyword.lower() in name.lower() for keyword in SENSITIVE_LAYER_KEYWORDS):
            sensitive_layers.append(name)

    func_1_0_print_colored(
        "\n# --- Potentially Sensitive Layers (Keyword-based Search) ---",
        AnsiColors.HEADER,
    )
    func_1_0_print_colored(
        "# Copy and paste these into 'custom_quantize_layers' in your .cfg file.",
        AnsiColors.OKCYAN,
    )

    if not sensitive_layers:
        func_1_0_print_colored(
            "# No layers matching the sensitive keywords found.", AnsiColors.WARNING
        )
        return

    for name in sorted(sensitive_layers):
        print(f"    '{name}': float16,")


def func_2_3_run_find_op_inputs(cfg_data, onnx_model, op_type):
    """
    Finds all inputs to a specific operator type in an ONNX model and
    matches them to layers in the CFG file.

    Args:
        cfg_data (dict): Parsed data from the config file.
        onnx_model (onnx.ModelProto): The loaded ONNX model.
        op_type (str): The operator type to search for (e.g., "Softmax").
    """
    func_1_0_print_colored(
        f"\n--- Finding CFG layers for ONNX operator type: {op_type} ---",
        AnsiColors.HEADER,
    )

    # Create a set for faster lookups
    cfg_layer_names = set(cfg_data.keys())

    found_nodes = 0
    matched_layers = set()

    # Iterate through all nodes in the ONNX graph
    for node in onnx_model.graph.node:
        if node.op_type == op_type:
            found_nodes += 1
            # A node can have multiple inputs
            for input_name in node.input:
                # Check if this input tensor name exists as a layer in the CFG
                if input_name in cfg_layer_names:
                    matched_layers.add(input_name)

    if found_nodes == 0:
        func_1_0_print_colored(
            f"Warning: No nodes of type '{op_type}' were found in the ONNX model.",
            AnsiColors.WARNING,
        )
        return

    func_1_0_print_colored(
        f"Found {found_nodes} '{op_type}' nodes in the ONNX model.", AnsiColors.OKBLUE
    )

    if not matched_layers:
        func_1_0_print_colored(
            f"Error: Found '{op_type}' nodes, but none of their input tensors matched any layer name in the CFG file.",
            AnsiColors.FAIL,
        )
        func_1_0_print_colored(
            "This could mean the CFG file and ONNX model are mismatched.",
            AnsiColors.WARNING,
        )
        return

    func_1_0_print_colored(
        f"\n# --- Matched Layers (for OP Type: {op_type}) ---", AnsiColors.HEADER
    )
    func_1_0_print_colored(
        "# Copy and paste these into 'custom_quantize_layers' in your .cfg file.",
        AnsiColors.OKCYAN,
    )

    for layer_name in sorted(list(matched_layers)):
        print(f"    '{layer_name}': float16,")


# ==============================================================================
# Level 3: Main Execution and Argument Parsing
# ==============================================================================


def func_3_0_setup_arg_parser():
    """
    Sets up and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: The configured argument parser object.
    """
    parser = argparse.ArgumentParser(
        description="Inspect RKNN .cfg files, with optional ONNX cross-referencing.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the .quantization.cfg file to inspect."
    )

    # --- Mode Selection ---
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="MODE 1: Show a summary of all operator types found in the CFG file.",
    )
    mode_group.add_argument(
        "-f",
        "--filter",
        nargs="+",
        metavar="KEYWORD",
        help="MODE 2: Filter and display layers from CFG containing one or more keywords.",
    )
    mode_group.add_argument(
        "-e",
        "--export-sensitive",
        action="store_true",
        help="MODE 3: (Basic) Export sensitive layers based on a predefined keyword list.",
    )
    mode_group.add_argument(
        "--find-inputs-for",
        type=str,
        metavar="OP_TYPE",
        help="MODE 4: (Advanced) Find layers in CFG that are inputs to a specific ONNX OP_TYPE (e.g., Softmax).",
    )

    # --- Additional Arguments for Advanced Modes ---
    parser.add_argument(
        "--onnx-model",
        type=str,
        metavar="PATH",
        help="Path to the corresponding .onnx model (required for --find-inputs-for mode).",
    )

    return parser


def main():
    """
    Main function to orchestrate the entire script execution.
    """
    # Level 3 call
    parser = func_3_0_setup_arg_parser()
    args = parser.parse_args()

    # --- Argument Validation ---
    if args.find_inputs_for and not args.onnx_model:
        parser.error("--onnx-model is required when using --find-inputs-for.")

    # --- Data Loading ---
    # Level 2 calls Level 1
    cfg_data = func_1_1_parse_cfg_file(args.config_file)
    if cfg_data is None:
        sys.exit(1)

    onnx_model = None
    if args.onnx_model:
        # Level 2 calls Level 1
        onnx_model = func_1_2_load_onnx_model(args.onnx_model)
        if onnx_model is None:
            sys.exit(1)

    # --- Mode Execution ---
    # Level 3 calls Level 2
    if args.summary:
        func_2_0_run_summary(cfg_data)
    elif args.filter:
        func_2_1_run_filter(cfg_data, args.filter)
    elif args.export_sensitive:
        func_2_2_run_export_sensitive(cfg_data)
    elif args.find_inputs_for:
        func_2_3_run_find_op_inputs(cfg_data, onnx_model, args.find_inputs_for)


if __name__ == "__main__":
    main()
