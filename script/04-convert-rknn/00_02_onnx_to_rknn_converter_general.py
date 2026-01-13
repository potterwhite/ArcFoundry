# Filename: unified_onnx_to_rknn_converter.py

import os
import sys
import onnx
import onnxsim
from rknn.api import RKNN
import numpy as np
import onnx_graphsurgeon as gs
import onnxruntime
from pathlib import Path
import threading
from queue import Queue
import argparse


def input_with_timeout(prompt, timeout, default="y"):
    """
    Prompts the user for input with a timeout.

    Args:
        prompt (str): The prompt to display to the user.
        timeout (int): The timeout in seconds.
        default (str): The default value to return if the timeout is reached.

    Returns:
        str: The user's input, or the default value if timed out.
    """
    # 在提示信息中加入超时和默认值说明，更友好
    full_prompt = f"{prompt} (timeout in {timeout}s, default: '{default}'): "

    q = Queue()

    def get_input_thread(queue_obj):
        try:
            # 在子线程中运行 input()
            user_input = input(full_prompt)
            queue_obj.put(user_input)
        except EOFError:
            # 如果输入流被关闭 (例如在管道中)，也放入一个标记
            queue_obj.put(None)

    input_thread = threading.Thread(target=get_input_thread, args=(q,))
    input_thread.daemon = True  # 设置为守护线程，这样主程序退出时它也会退出
    input_thread.start()

    try:
        # 等待子线程，但有超时
        user_input = q.get(timeout=timeout)
        # 如果能获取到，说明用户在超时前输入了
        return user_input
    except:
        # 如果 q.get() 超时 (抛出 queue.Empty 异常) 或发生其他错误
        print(f"\n[INFO] Timeout reached. Proceeding with default value '{default}'.")
        return default


# --- 如何替换您原来的 ask_to_continue 函数 ---


def ask_to_continue(prompt, timeout=10):
    """
    Displays a prompt and asks the user to continue or exit, with a timeout.
    Defaults to 'yes' on timeout.

    Args:
        prompt (str): The prompt to display.
        timeout (int): Timeout in seconds. Defaults to 10.

    Returns:
        bool: True to continue, False to exit.
    """
    # 调用我们新的带超时的 input 函数
    # 注意，我们把 [Y/n] 的提示移到了 input_with_timeout 内部
    response = (
        input_with_timeout(f"❓ {prompt}", timeout=timeout, default="y").lower().strip()
    )

    # 后续逻辑保持不变
    if response in ["y", "yes", ""]:
        return True
    elif response in ["n", "no"]:
        print("❌ Operation cancelled by user.")
        return False
    else:
        # 如果超时，response 会是 'y'，会走上面的 if 分支
        # 只有在用户输入了无效内容时才会到这里
        print("Invalid input. Proceeding with default (Yes).")
        return True


class ConversionConfig:
    """
    A dedicated class to hold all user-configurable settings.
    This acts as a clean and explicit 'CONTROL PANEL' for the script.
    """

    def __init__(self):
        PY_SCRIPT_PATH = Path(__file__).resolve()
        PY_SCRIPT_DIR = PY_SCRIPT_PATH.parent

        # 1. Set the target hardware platform
        self.PLATFORM = "rv1126b"

        # 2. Enable or disable quantization.
        # self.QUANTIZATION_ENABLED = True
        self.QUANTIZATION_ENABLED = False

        # 3. Set the optimization level (0 to 3).
        self.OPTIMIZATION_LEVEL = 3  # <-- 恢复默认优化等级
        # self.OPTIMIZATION_LEVEL = 0

        # 4. Set the data type for quantization.
        self.QUANTIZED_TYPE = "asymmetric_quantized-8"

        # 5. Path to the dataset file for quantization calibration.
        # self.DATASET_PATH_FOR_QUANT = f"{PY_SCRIPT_DIR}/../dataset/dataset.txt"
        # self.DATASET_PATH_FOR_QUANT = f"{PY_SCRIPT_DIR}/../dataset/dataset_streaming.txt"
        # self.DATASET_PATH_FOR_QUANT = (
        #    f"{PY_SCRIPT_DIR}/../../dataset/dataset_snapshot.txt"
        # )
        # self.DATASET_PATH_FOR_QUANT = (
        #    f"{PY_SCRIPT_DIR}/../../dataset/dataset_hybrid.txt"
        # )
        self.DATASET_PATH_FOR_QUANT = (
            f"{PY_SCRIPT_DIR}/../../dataset/dataset_v09_interval_1.txt"
        )

        # 6. Define where to save the output and intermediate models.
        self.BASE_MODEL_DIR = f"{PY_SCRIPT_DIR}/../../models"
        self.ONNX_FIXED_DIR = os.path.join(self.BASE_MODEL_DIR, "onnx_fixed")
        self.RKNN_OUTPUT_DIR = os.path.join(self.BASE_MODEL_DIR, "rknn")

        # 7. verbose flag
        self.VERBOSE_ENABLED = True

        # 8. Perf flag
        self.RKNN_PERF_DEBUG_ENABLED = True
        # self.RKNN_PERF_DEBUG_ENABLED = False

        # 9. eval memory flag
        self.RKNN_EVAL_MEMORY_ENABLED = True
        # self.RKNN_EVAL_MEMORY_ENABLED = False

        # 10. model pruning flag
        self.RKNN_PRUNING_ENABLED = True
        # self.RKNN_PRUNING_ENABLED = False

        # 11. Define a template path for the quantization configuration file.
        #    The {model_name} will be filled in dynamically.
        self.QUANT_CONFIG_PATH_TEMPLATE = os.path.join(
            self.RKNN_OUTPUT_DIR, "{model_name}_quant_config.txt"
        )


class OnnxModelConverter:
    """
    Encapsulates all logic for converting a single ONNX model to RKNN.
    Follows a robust two-step process:
    1. Create a fixed-size intermediate ONNX model.
    2. Convert the fixed-size ONNX model to RKNN.
    """

    # (In class OnnxModelConverter)

    # ==================== REPLACE YOUR __init__ METHOD WITH THIS FINAL VERSION ====================
    def __init__(
        self,
        model_name,
        onnx_path,
        config,
        output_dir,
        export_quant_config=False,
        import_quant_config=False,
        auto_hybrid=False,
    ):
        """
        Initializes the converter for a single ONNX model.
        """
        self.model_name = model_name
        self.original_onnx_path = onnx_path
        self.config = config
        self.output_dir = output_dir

        # Store the operational mode flags
        self.export_quant_config = export_quant_config
        self.import_quant_config = import_quant_config
        self.auto_hybrid = auto_hybrid

        self.verbose_enabled = self.config.VERBOSE_ENABLED

        # Define paths for intermediate and final files
        self.fixed_onnx_path = os.path.join(
            self.config.ONNX_FIXED_DIR, f"{self.model_name}_fixed.onnx"
        )
        self.simplified_onnx_path = os.path.join(
            self.config.ONNX_FIXED_DIR, f"{self.model_name}_simplified.onnx"
        )
        self.rknn_path = os.path.join(self.output_dir, f"{self.model_name}.rknn")

        # Path for the manual quantization config file
        self.quant_config_path = self.config.QUANT_CONFIG_PATH_TEMPLATE.format(
            model_name=self.model_name
        )

    # In OnnxModelConverter.convert()
    def convert(self):
        """
        Public method to run the full conversion pipeline for one model.
        """
        print(
            f"\n================= Processing Model: {self.model_name.upper()} =================\n"
        )

        onnx_for_rknn_conversion = self.original_onnx_path

        # --- THIS IS THE FINAL, CORRECT LOGIC ---
        # Check if the current model is one of the main, dynamic-shape models
        is_main_model = any(
            name in self.model_name.lower() for name in ["encoder", "decoder", "joiner"]
        )

        if is_main_model:
            print(
                f"--- Model '{self.model_name}' requires standard processing (shape fixing, simplifying). ---"
            )
            if not self._create_fixed_size_onnx():
                return False

            # NOTE: Your original script simplifies all main models. We keep this logic.
            if not self._simplify_decoder_model():
                return False

            onnx_for_rknn_conversion = self.simplified_onnx_path

            prompt_text = (
                f"Intermediate model for '{self.model_name}' created. Continue?"
            )
            if not ask_to_continue(prompt_text):
                return False
        else:
            # This branch is for special, pre-processed models like 'attention_block'
            print(
                f"--- Model '{self.model_name}' is a pre-processed submodule. Skipping all intermediate steps. ---"
            )
            # We directly use the original path, no modifications needed.
        # ---------------------------------------------

        # The rest of the function (inspect metadata, run rknn conversion) remains the same.
        custom_string = self._inspect_metadata()
        prompt_text = f"meta data of model '{self.model_name}' has been inspected. Continue to RKNN conversion?"
        if not ask_to_continue(prompt_text):
            return False

        return self._run_rknn_conversion(
            custom_string, onnx_for_rknn_conversion, self.rknn_path
        )

    def _create_fixed_size_onnx(self):
        """
        Loads a dynamic-input ONNX model and saves a new version
        with all dynamic input dimensions (e.g., batch size 'N') fixed to 1.
        """
        print(f"--- Step 1: Creating fixed-size ONNX model ---")
        print(f"Input: {self.original_onnx_path}")
        print(f"Output: {self.fixed_onnx_path}")

        try:
            os.makedirs(os.path.dirname(self.fixed_onnx_path), exist_ok=True)
            model = onnx.load(self.original_onnx_path)

            for model_input in model.graph.input:
                # For decoder, also fix the data type to INT64
                if "decoder" in self.model_name.lower() and model_input.name == "y":
                    model_input.type.tensor_type.elem_type = onnx.TensorProto.INT64

                shape = model_input.type.tensor_type.shape
                if shape:
                    print(
                        f"  - Checking input '{model_input.name}': Original shape {[d.dim_param if d.dim_param else d.dim_value for d in shape.dim]}"
                    )
                    for dim in shape.dim:
                        if dim.dim_param:
                            print(
                                f"    - Fixing dynamic dimension '{dim.dim_param}' to 1."
                            )
                            dim.ClearField("dim_param")
                            dim.dim_value = 1

            onnx.save(model, self.fixed_onnx_path)
            print("--- Successfully created fixed-size ONNX model. ---")
            return True

        except Exception as e:
            print(
                f"[ERROR] Failed to create fixed-size ONNX model: {e}", file=sys.stderr
            )
            return False

    # <-- 4. 添加新的私有方法用于简化decoder
    def _simplify_decoder_model(self):
        """
        Uses onnx-simplifier to prune static branches from the fixed-size decoder model.
        """
        print(f"\n--- Step 1.5: Simplifying decoder model with onnx-simplifier ---")
        print(f"Input (Fixed ONNX): {self.fixed_onnx_path}")
        print(f"Output (Simplified ONNX): {self.simplified_onnx_path}")
        try:
            model = onnx.load(self.fixed_onnx_path)
            model_simplified, check = onnxsim.simplify(model)
            assert check, "onnx-simplifier failed to simplify the model."
            onnx.save(model_simplified, self.simplified_onnx_path)
            print("--- Successfully simplified decoder model. ---")
            return True
        except Exception as e:
            print(
                f"[ERROR] An error occurred during ONNX simplification: {e}",
                file=sys.stderr,
            )
            return False

    def _inspect_metadata(self):
        """
        Inspects the ORIGINAL ONNX model to extract the custom metadata map.
        """
        print(f"\n--- Step 2: Inspecting metadata from original ONNX ---")
        try:
            session = onnxruntime.InferenceSession(
                self.original_onnx_path, providers=["CPUExecutionProvider"]
            )
            meta = session.get_modelmeta()
            custom_metadata = meta.custom_metadata_map

            if not custom_metadata:
                print("\n[INFO] No custom metadata found in this ONNX model.")
                return ""  # Return empty string if no metadata

            print("\n--- Found Custom Metadata (RKNN custom_string source) ---")
            for key, value in custom_metadata.items():
                print(f"  - {key}: {value}")

            reconstructed_string = ";".join(
                [f"{key}={value}" for key, value in custom_metadata.items()]
            )
            print("\n--- Reconstructed 'custom_string' for RKNN ---")
            print(reconstructed_string)
            return reconstructed_string

        except Exception as e:
            print(
                f"\n[ERROR] Failed to read or parse the ONNX model: {e}",
                file=sys.stderr,
            )
            return ""  # Return empty string on error

    # (In class OnnxModelConverter)
    # ==================== REPLACE THIS FUNCTION WITH THE FINAL, ROBUST VERSION ====================
    def _run_rknn_conversion(self, custom_string, onnx_path, rknn_path):
        """
        Performs the conversion. This version correctly handles config parameters for non-quantized models.
        """
        do_quantization = (
            self.config.QUANTIZATION_ENABLED and "encoder" in self.model_name
        )

        print(f"\n--- Step 3: Performing RKNN Conversion ---")

        rknn = RKNN(verbose=self.verbose_enabled)

        # --- Mode 1: Manual Hybrid Step 1 (Export Config) ---
        if self.export_quant_config:
            # ... (This part is correct, no changes needed)
            print("[INFO] Mode: Manual Hybrid - Step 1 (Exporting quantization config)")
            if not do_quantization:
                print(
                    "[ERROR] --export-quant-config is only for models to be quantized.",
                    file=sys.stderr,
                )
                rknn.release()
                return False
            rknn.config(
                target_platform=self.config.PLATFORM, custom_string=custom_string
            )
            if rknn.load_onnx(model=onnx_path) != 0:
                print(
                    f"[ERROR] Failed to load ONNX model: {onnx_path}", file=sys.stderr
                )
                rknn.release()
                return False
            if (
                rknn.hybrid_quantization_step1(
                    dataset=self.config.DATASET_PATH_FOR_QUANT
                )
                != 0
            ):
                print("[ERROR] hybrid_quantization_step1 failed.", file=sys.stderr)
                rknn.release()
                return False
            print("\n--- Successfully generated manual hybrid quantization files. ---")
            rknn.release()
            return True

        # --- Mode 2: Manual Hybrid Step 2 (Import Config) ---
        elif self.import_quant_config:
            # ... (This part is correct, no changes needed)
            print("[INFO] Mode: Manual Hybrid - Step 2 (Importing quantization config)")
            model_name_base = self.model_name + "_simplified"
            model_input, data_input, model_quantization_cfg = (
                f"./{model_name_base}.model",
                f"./{model_name_base}.data",
                f"./{model_name_base}.quantization.cfg",
            )
            for f in [model_input, data_input, model_quantization_cfg]:
                if not os.path.exists(f):
                    print(f"[ERROR] Required file not found: {f}", file=sys.stderr)
                    rknn.release()
                    return False
            if (
                rknn.hybrid_quantization_step2(
                    model_input=model_input,
                    data_input=data_input,
                    model_quantization_cfg=model_quantization_cfg,
                )
                != 0
            ):
                print("[ERROR] hybrid_quantization_step2 failed.", file=sys.stderr)
                rknn.release()
                return False

        # --- Mode 3 (Auto Hybrid) & Mode 4 (Normal Build) ---
        else:
            if self.auto_hybrid:
                print("[INFO] Mode: Auto Hybrid Quantization Build")
                if not do_quantization:
                    print(
                        "[ERROR] --auto-hybrid is only for models to be quantized.",
                        file=sys.stderr,
                    )
                    rknn.release()
                    return False
            else:
                print("[INFO] Mode: Normal RKNN Build Process")

            print("--> Configuring model...")
            # --- CRITICAL FIX: Dynamically build the config dictionary ---
            config_params = {
                "target_platform": self.config.PLATFORM,
                "custom_string": custom_string,
                "optimization_level": self.config.OPTIMIZATION_LEVEL,
                "model_pruning": self.config.RKNN_PRUNING_ENABLED,
            }
            if do_quantization:
                config_params["quantized_dtype"] = self.config.QUANTIZED_TYPE

            rknn.config(**config_params)
            # --- END OF FIX ---

            print(f"--> Loading ONNX model {onnx_path}...")
            if rknn.load_onnx(model=onnx_path) != 0:
                print(
                    f"[ERROR] Failed to load ONNX model: {onnx_path}", file=sys.stderr
                )
                rknn.release()
                return False

            print(f"--> Building model...")
            dataset = self.config.DATASET_PATH_FOR_QUANT if do_quantization else None
            if (
                rknn.build(
                    do_quantization=do_quantization,
                    dataset=dataset,
                    auto_hybrid=self.auto_hybrid,
                )
                != 0
            ):
                print(
                    f"[ERROR] Failed to build model from: {onnx_path}", file=sys.stderr
                )
                rknn.release()
                return False

        # --- Common Export and Evaluate Steps ---
        print(f"--> Exporting to RKNN model: {rknn_path}")
        os.makedirs(os.path.dirname(rknn_path), exist_ok=True)
        if rknn.export_rknn(rknn_path) != 0:
            print(f"[ERROR] Failed to export RKNN model: {rknn_path}", file=sys.stderr)
            rknn.release()
            return False

        if self.config.RKNN_PERF_DEBUG_ENABLED or self.config.RKNN_EVAL_MEMORY_ENABLED:
            print(f"--> Initializing runtime for performance evaluation...")
            if (
                rknn.init_runtime(
                    target=self.config.PLATFORM,
                    perf_debug=self.config.RKNN_PERF_DEBUG_ENABLED,
                    eval_mem=self.config.RKNN_EVAL_MEMORY_ENABLED,
                )
                != 0
            ):
                print(
                    f"[WARNING] init_runtime failed! (This is expected if not running on a device or if device is mismatched)"
                )
            else:
                print("--> Evaluating performance...")
                print(rknn.eval_perf(is_print=False))
                print("--> Evaluating memory...")
                print(rknn.eval_memory())

        print(f"--> Releasing RKNN context.")
        rknn.release()
        print("--- Conversion successful! ---")
        return True


# (In file 00_02_onnx_to_rknn_converter_general.py)
# ==================== REPLACE YOUR main() WITH THIS FINAL VERSION ====================
def main():
    """
    Main function to parse command-line arguments and orchestrate the model conversion process.
    """
    parser = argparse.ArgumentParser(
        description="A script to convert ONNX models to RKNN, with advanced hybrid quantization support.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "models",
        nargs="+",
        help="A space-separated list of models to convert (e.g., encoder).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the directory where RKNN models and config files will be saved.",
    )

    # --- HYBRID QUANTIZATION MODES ---
    hybrid_quant_group = parser.add_argument_group(
        "Hybrid Quantization Modes (Choose One)"
    )
    mode_group = hybrid_quant_group.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--export-quant-config",
        action="store_true",
        help="MODE 1: Run step1 to generate a manual quantization config file.",
    )
    mode_group.add_argument(
        "--import-quant-config",
        action="store_true",
        help="MODE 2: Run step2 to build a model using a manual quantization config file.",
    )
    mode_group.add_argument(
        "--auto-hybrid",
        action="store_true",
        help="MODE 3: (Recommended) Use the built-in auto hybrid quantization feature.",
    )
    # ---------------------------------------------

    args = parser.parse_args()
    config = ConversionConfig()

    final_output_dir = args.output
    print(f"[INFO] Using output directory: {final_output_dir}")
    os.makedirs(final_output_dir, exist_ok=True)

    models_to_process = {}
    default_onnx_dir = os.path.join(config.BASE_MODEL_DIR, "onnx")
    for model_arg in args.models:
        model_name = (
            Path(model_arg).stem.replace("-epoch-99-avg-1", "")
            if os.path.isfile(model_arg)
            else model_arg
        )
        model_path = (
            model_arg
            if os.path.isfile(model_arg)
            else os.path.join(default_onnx_dir, f"{model_name}-epoch-99-avg-1.onnx")
        )
        models_to_process[model_name] = model_path

    print("\nThe following models will be processed:")
    for name, path in models_to_process.items():
        print(f"  - {name}: {path}")

    all_conversions_succeeded = True
    for name, path in models_to_process.items():
        if not os.path.exists(path):
            print(
                f"\n[FATAL] Input ONNX file not found for '{name}': {path}",
                file=sys.stderr,
            )
            all_conversions_succeeded = False
            break

        converter = OnnxModelConverter(
            model_name=name,
            onnx_path=path,
            config=config,
            output_dir=final_output_dir,
            export_quant_config=args.export_quant_config,
            import_quant_config=args.import_quant_config,
            auto_hybrid=args.auto_hybrid,
        )

        if not converter.convert():
            all_conversions_succeeded = False
            print(
                f"\n[FATAL] Operation failed for '{name}'. Stopping script.",
                file=sys.stderr,
            )
            break

    print("\n========================================================")
    if all_conversions_succeeded:
        print(f"✅ All specified operations completed successfully!")
        print(f"   Outputs are in: {final_output_dir}")
    else:
        print("❌ One or more operations failed.")
    print("========================================================")


if __name__ == "__main__":
    main()
