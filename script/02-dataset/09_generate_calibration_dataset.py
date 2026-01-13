# 09_generate_calibration_dataset.py

import os
import numpy as np
import librosa
from pathlib import Path
import onnxruntime as ort
import sys
import logging
from tqdm import tqdm


# ====================================================================================
# Func 1.0: Configuration Area
# ====================================================================================
def func_1_0_setup_configuration():
    """
    Sets up all the necessary paths, model names, and parameters for the script.
    """
    # --- CRITICAL: ADJUST THIS VALUE TO CONTROL DATASET SIZE ---
    SAMPLING_INTERVAL = 1    # ->  Saves every step (original 05.py behavior, ~3 hours quantization)
    # SAMPLING_INTERVAL = 2      # ->  Saves every 2nd step (~1.5 hours quantization)
    # SAMPLING_INTERVAL = 10   # ->  Saves every 10th step (~20 mins quantization)
    # SAMPLING_INTERVAL = 4    # <--- 在这里控制采样的间隔

    # --- Paths and Files ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    ONNX_MODEL_DIR = SCRIPT_DIR / ".." / ".." / "models" / "onnx"
    DATASET_DIR = SCRIPT_DIR / ".." / ".." / "dataset"

    # Updated naming convention as requested
    CALIBRATION_NPY_DIR = (
        DATASET_DIR / f"calibration_npy_v09_interval_{SAMPLING_INTERVAL}"
    )
    FINAL_DATASET_FILE = DATASET_DIR / f"dataset_v09_interval_{SAMPLING_INTERVAL}.txt"

    AUDIO_CANDIDATES_FILE = DATASET_DIR / "calibration_candidates.txt"

    config = {
        "sampling_interval": SAMPLING_INTERVAL,
        "calibration_npy_dir": CALIBRATION_NPY_DIR,
        "final_dataset_file": FINAL_DATASET_FILE,
        "audio_candidates_file": AUDIO_CANDIDATES_FILE,
        "encoder_onnx": ONNX_MODEL_DIR / "encoder-epoch-99-avg-1.onnx",
        "sample_rate": 16000,
        "n_fft": 400,
        "hop_length": 160,
        "win_length": 400,
        "n_mels": 80,
        "chunk_size": 39,
        "chunk_shift": 19,
        # Parameters from sherpa-onnx/csrc/features.h for accurate feature extraction
        "dither": 0.0,
        "remove_dc_offset": True,
        "preemph_coeff": 0.97,
    }
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return config


# ====================================================================================
# Func 1.1: Feature Extraction (IMPROVED - from script 08)
# ====================================================================================
def func_1_1_extract_features(waveform, config):
    """
    Extracts features from an audio waveform using the sherpa-onnx aligned pipeline.
    """
    try:
        if config["dither"] > 0.0:
            waveform += config["dither"] * np.random.randn(*waveform.shape)
        if config["remove_dc_offset"]:
            waveform -= np.mean(waveform)
        if config["preemph_coeff"] > 0.0:
            waveform = librosa.effects.preemphasis(
                waveform, coef=config["preemph_coeff"]
            )

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=config["sample_rate"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            win_length=config["win_length"],
            n_mels=config["n_mels"],
        )
        log_mel = np.log(np.maximum(mel, 1e-10))

        return log_mel.T.astype(np.float32)
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        return None


# ====================================================================================
# Func 1.2: ONNX Model Loader
# ====================================================================================
def func_1_2_load_onnx_encoder(config):
    def get_numpy_dtype(onnx_type_str):
        type_map = {
            "tensor(float)": "float32",
            "tensor(int64)": "int64",
            "tensor(float16)": "float16",
            "tensor(double)": "float64",
            "tensor(int8)": "int8",
            "tensor(int16)": "int16",
            "tensor(int32)": "int32",
            "tensor(uint8)": "uint8",
            "tensor(uint16)": "uint16",
            "tensor(uint32)": "uint32",
            "tensor(uint64)": "uint64",
            "tensor(bool)": "bool",
        }
        return type_map.get(onnx_type_str)

    try:
        session = ort.InferenceSession(str(config["encoder_onnx"]))
        inputs_info = [
            {"name": i.name, "shape": i.shape, "type": get_numpy_dtype(i.type)}
            for i in session.get_inputs()
        ]
        outputs_info = [o.name for o in session.get_outputs()]
        return session, inputs_info, outputs_info
    except Exception as e:
        logging.error(f"Failed to load ONNX encoder model: {e}")
        sys.exit(1)


# ====================================================================================
# Func 1.3: Single Audio Processor (MODIFIED with Systematic Sampling)
# ====================================================================================
def func_1_3_process_single_audio(
    audio_path, session, inputs_info, outputs_info, initial_states, config
):
    try:
        waveform, sr = librosa.load(audio_path, sr=config["sample_rate"])
    except Exception as e:
        logging.error(f"Failed to load audio {audio_path}: {e}")
        return []

    all_features = func_1_1_extract_features(waveform, config)
    if all_features is None:
        return []

    base_filename = Path(audio_path).stem
    total_frames = all_features.shape[0]
    current_states = initial_states
    generated_lines = []
    step_counter = 0

    for start_frame in range(0, total_frames, config["chunk_shift"]):
        end_frame = start_frame + config["chunk_size"]
        if end_frame > total_frames:
            break

        step_counter += 1

        # --- Systematic Sampling Logic ---
        if (step_counter - 1) % config["sampling_interval"] == 0:
            feature_chunk_to_save = np.expand_dims(
                all_features[start_frame:end_frame, :], axis=0
            )
            onnx_inputs_to_save = {
                inputs_info[0]["name"]: feature_chunk_to_save,
                **current_states,
            }

            current_data_line_paths = []
            step_dir = (
                config["calibration_npy_dir"] / f"{base_filename}_step_{start_frame}"
            )
            step_dir.mkdir(parents=True, exist_ok=True)
            for name, data in onnx_inputs_to_save.items():
                npy_path = step_dir / f"{name.replace('/', '_')}.npy"
                np.save(npy_path, data)
                current_data_line_paths.append(str(npy_path.resolve()))
            generated_lines.append(" ".join(current_data_line_paths))

        # Inference MUST run on EVERY step to get correct states for the next step.
        try:
            inference_feature_chunk = np.expand_dims(
                all_features[start_frame:end_frame, :], axis=0
            )
            onnx_inputs_for_inference = {
                inputs_info[0]["name"]: inference_feature_chunk,
                **current_states,
            }
            outputs = session.run(outputs_info, onnx_inputs_for_inference)

            new_states = {}
            for i, output_name in enumerate(outputs_info[1:]):
                state_input_name = inputs_info[i + 1]["name"]
                new_states[state_input_name] = outputs[i + 1]
            current_states = new_states
        except Exception as e:
            logging.warning(
                f"ONNX inference failed for chunk at {start_frame} in {audio_path}: {e}"
            )
            break

    return generated_lines


# ====================================================================================
# Main Execution Logic
# ====================================================================================
def main():
    config = func_1_0_setup_configuration()
    logging.info(
        f"Configuration: Systematic sampling with interval {config['sampling_interval']}"
    )

    session, inputs_info, outputs_info = func_1_2_load_onnx_encoder(config)

    try:
        with open(config["audio_candidates_file"], "r") as f:
            audio_paths = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(
            f"Audio candidates file not found: {config['audio_candidates_file']}"
        )
        sys.exit(1)

    config["calibration_npy_dir"].mkdir(parents=True, exist_ok=True)

    initial_states = {
        info["name"]: np.zeros(
            [1 if isinstance(d, str) else d for d in info["shape"]],
            dtype=np.dtype(info["type"]),
        )
        for info in inputs_info[1:]
    }

    all_dataset_lines = []
    for audio_path in tqdm(audio_paths, desc="Processing Audios"):
        lines = func_1_3_process_single_audio(
            audio_path, session, inputs_info, outputs_info, initial_states, config
        )
        all_dataset_lines.extend(lines)

    if all_dataset_lines:
        with open(config["final_dataset_file"], "w") as f:
            for line in all_dataset_lines:
                f.write(line + "\n")
        logging.info(
            f"Successfully generated {len(all_dataset_lines)} lines of calibration data."
        )
        logging.info(f"Final dataset file saved to: {config['final_dataset_file']}")
    else:
        logging.warning("No calibration data was generated.")


if __name__ == "__main__":
    main()
