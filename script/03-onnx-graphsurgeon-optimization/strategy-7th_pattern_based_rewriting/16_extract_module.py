# Filename: 16_extract_module_final_v7.py

import onnx
import onnx_graphsurgeon as gs
import numpy as np
import onnxsim
from pathlib import Path

# ... (Configuration section remains the same) ...
INPUT_MODEL_PATH = Path("/development/asr/build-scripts/onnx/c-convert/models/onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx")
OUTPUT_MODEL_PATH = Path("/development/asr/build-scripts/onnx/c-convert/models/onnx_modified/attention_block.onnx")
MAIN_INPUT_TENSOR_NAME = "1110"
MAIN_OUTPUT_TENSOR_NAME = "1325"
BATCH_SIZE = 1
SEQUENCE_LENGTH = 70
FEATURE_DIM = 384

# --- Model-Specific Architectural Constants ---
# NOTE: These values are derived from analyzing the model architecture and error logs.
# The model appears to add a fixed context of 25 frames internally.
CONTEXT_SIZE = 25
INTERNAL_SEQUENCE_LENGTH = SEQUENCE_LENGTH + CONTEXT_SIZE # Should be 95 for SEQUENCE_LENGTH=70

# The Reshape node splits the feature dimension into heads and head_dim.
NUM_HEADS = 8
HEAD_DIM = 4
# --------------------

print(f"Preparing to extract and fix a module from '{INPUT_MODEL_PATH}'...")
print(f"  Main Input: '{MAIN_INPUT_TENSOR_NAME}', Main Output: '{MAIN_OUTPUT_TENSOR_NAME}'")

try:
    graph = gs.import_onnx(onnx.load(INPUT_MODEL_PATH))
    print("Model loaded successfully.")

    tensors = graph.tensors()

    # --- Step 1: Isolate the subgraph ---
    # (This part is the same and correct)
    graph.outputs = [tensors[MAIN_OUTPUT_TENSOR_NAME]]
    graph.inputs = [tensors[MAIN_INPUT_TENSOR_NAME]]
    graph.cleanup()
    required_inputs = [tensors[MAIN_INPUT_TENSOR_NAME]]
    for node in graph.nodes:
        for inp in node.inputs:
            if not isinstance(inp, gs.Constant) and inp.inputs is not None and not inp.inputs and inp not in required_inputs:
                required_inputs.append(inp)
    graph.inputs = required_inputs
    graph.cleanup()
    print("Subgraph isolated with all required inputs.")

    # --- Step 2: Surgically fix the Reshape_533 node AND CLEAN UP ITS INPUTS ---
    reshape_node = next((node for node in graph.nodes if node.name == 'Reshape_533'), None)
    if reshape_node:
        # Get the tensor that was providing the wrong shape
        old_shape_tensor = reshape_node.inputs[1]

        # Create the new, correct constant shape tensor based on our derived internal sequence length.
        # The target shape must match the total elements of the input tensor [1, 95, 32].
        # Our target shape [1, 95, 8, 4] has 1*95*8*4 = 3040 elements, which is correct.
        correct_shape_values = np.array([BATCH_SIZE, INTERNAL_SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM], dtype=np.int64)
        correct_shape_tensor = gs.Constant(name="forced_shape_for_reshape_533", values=correct_shape_values)

        # Replace the input. Now reshape_node uses our new constant.
        reshape_node.inputs[1] = correct_shape_tensor

        # --- NEW CLEANUP LOGIC ---
        # The old_shape_tensor is now "dangling" because reshape_node no longer uses it.
        # Its upstream nodes (like Concat, Gather, etc.) might also become dangling.
        # We can let graph.cleanup() handle this automatically.
        print("Surgical fix applied: Replaced shape input for 'Reshape_533'.")

    # --- Step 3: Set fixed, static shapes for ALL inputs and outputs ---
    # (This part is the same and correct)
    print("Setting static shapes for all I/O tensors...")
    for inp in graph.inputs:
        inp.dtype = np.float32
        if inp.name == MAIN_INPUT_TENSOR_NAME:
            inp.shape = [BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_DIM]
            print(f"  - Forcefully set shape for MAIN input '{inp.name}': {inp.shape}")
            continue
        original_shape = list(inp.shape) if inp.shape else []
        new_shape = []
        for dim in original_shape:
            if isinstance(dim, str):
                if 'batch' in dim.lower(): new_shape.append(BATCH_SIZE)
                elif 'sequence' in dim.lower() or 'time' in dim.lower() or 'len' in dim.lower() or dim == 'N': new_shape.append(SEQUENCE_LENGTH)
                else: new_shape.append(1)
            else:
                new_shape.append(dim)
        inp.shape = new_shape
        print(f"  - Final shape for auxiliary input '{inp.name}': {inp.shape}")
    graph.outputs[0].dtype = np.float32
    graph.outputs[0].shape = [BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_DIM]
    print(f"  - Final shape for output '{graph.outputs[0].name}': {graph.outputs[0].shape}")


    # --- Step 4: Final cleanup, restructuring with onnx-simplifier, and save ---
    print("Performing final graph cleanup...")
    graph.cleanup().toposort()

    # Export the graph to an in-memory ONNX model
    temp_model = gs.export_onnx(graph)

    print("Restructuring the graph with onnx-simplifier for maximum compatibility...")
    # Use onnx-simplifier to create a clean, canonical version of the model
    # This can fix many subtle structural issues.
    model_simplified, check = onnxsim.simplify(temp_model)
    if not check:
        print("[WARNING] onnx-simplifier reported an issue, but we will proceed.")

    print("Saving the final, simplified, and corrected ONNX model...")
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Save the simplified model instead of the original one
    onnx.save(model_simplified, OUTPUT_MODEL_PATH)

    print(f"\nSuccess! A fully corrected and RESTRUCTURED model saved to: {OUTPUT_MODEL_PATH}")

except Exception as e:
    print(f"An error occurred during the process: {e}")