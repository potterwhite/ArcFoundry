import onnxruntime as ort
import numpy as np
from pathlib import Path


ORIGINAL_MODEL_PATH = Path("/development/asr/build-scripts/onnx/c-convert/models/onnx/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx")
OPTIMIZED_MODEL_PATH = Path("/development/asr/build-scripts/onnx/c-convert/models/onnx_modified/transpose_inverse.onnx")

DTYPE_MAP = {
    'tensor(float)': np.float32,
    'tensor(float16)': np.float16,
    'tensor(double)': np.float64,
    'tensor(int8)': np.int8,
    'tensor(int16)': np.int16,
    'tensor(int32)': np.int32,
    'tensor(int64)': np.int64,
    'tensor(bool)': np.bool_,

}

def verify_models(original_model, optimized_model):
    print("--- starting numerical verification ---")
    try:
        print(f"Loading original model: {original_model}")
        sess_orig = ort.InferenceSession(str(original_model))
        print(f"Loading optimized model: {optimized_model}")
        sess_new = ort.InferenceSession(str(optimized_model))
        print(f"Loading models successfully.")


        input_meta = sess_orig.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape

        static_shape = [1 if not isinstance(dim, int) else dim for dim in input_shape]

        print("\nOriginal Model Inputs:")
        model_inputs = sess_orig.get_inputs()
        for i, inp in enumerate(model_inputs):
            print(f"  Input {i}: name='{inp.name}', shape={inp.shape}, dtype={inp.type}")

        input_name = sess_orig.get_inputs()[0].name

        sample_shape = (1, 39, 80)
        print(f"\nCreating a random input tensor with shape: {sample_shape}")
        input_data = np.random.rand(*sample_shape).astype(np.float32)

        inputs_dict = {}
        for inp in model_inputs:
            if inp.name == input_name:
                inputs_dict[inp.name] = input_data
            else:
                other_shape = [1 if not isinstance(dim, int) else dim for dim in inp.shape]
                correct_dtype = DTYPE_MAP.get(inp.type, np.float32)
                inputs_dict[inp.name] = np.zeros(other_shape, dtype=correct_dtype)

        print("running inference on original model...")
        outputs_orig = sess_orig.run(None, inputs_dict)

        print("running inference on optimized model...")
        outputs_new = sess_new.run(None, inputs_dict)
        print("inference complete.")

        # compare the result
        print("\ncomparing outputs...")
        if len(outputs_orig) != len(outputs_new):
            print(f"[Fail] mismatch in number of outputs! original: {len(outputs_orig)}, optimized: {outputs_new}")
            return

        all_match = True
        for i, (out_orig, out_new) in enumerate(zip(outputs_orig, outputs_new)):
            try:
                np.testing.assert_allclose(out_orig, out_new, rtol=1e-3, atol=1e-3)
                print(f"  - output {i}: match (shape: {out_orig.shape})")
            except AssertionError as e:
                print(f"  - output {i}: mismatch! (shape: {out_orig.shape})")
                print(e)
                all_match = False

        if all_match:
            print("\n[success] All outputs match! the optimization is numerically correct.")
        else:
            print("\n[Fail] some outputs do not match. the optimization introduced errors.")


    except Exception as e:
        print(f"\n[Error] An error occurred during verification: {e}")

def main():
    verify_models(ORIGINAL_MODEL_PATH, OPTIMIZED_MODEL_PATH)

if __name__ == "__main__":
    main()


























