# 05_generate_calibration_dataset_streaming.py

import os
import numpy as np
import librosa
from pathlib import Path
import onnxruntime as ort
import sys
import logging

# ====================================================================================
# Func 1.0: Configuration Area
# Description: This area defines all global configurations and constants for the script.
# ====================================================================================
def func_1_0_setup_configuration():
    """
    Sets up all the necessary paths, model names, and parameters for the script.
    """
    # --- Paths and Files ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    ONNX_MODEL_DIR = SCRIPT_DIR / ".." / "models" / "onnx"
    DATASET_DIR = SCRIPT_DIR / ".." / "dataset"
    CALIBRATION_NPY_DIR = DATASET_DIR / "calibration_npy_streaming"
    FINAL_DATASET_FILE = DATASET_DIR / "dataset_streaming.txt"
    AUDIO_CANDIDATES_FILE = DATASET_DIR / "calibration_candidates.txt"

    # --- Model File Names ---
    ENCODER_ONNX_FILENAME = "encoder-epoch-99-avg-1.onnx"
    DECODER_ONNX_FILENAME = "decoder-epoch-99-avg-1.onnx"
    JOINER_ONNX_FILENAME = "joiner-epoch-99-avg-1.onnx"

    # --- Fbank Feature Extraction Parameters ---
    # These parameters must match the sherpa-onnx runtime exactly.
    config = {
        'script_dir': SCRIPT_DIR,
        'onnx_model_dir': ONNX_MODEL_DIR,
        'calibration_npy_dir': CALIBRATION_NPY_DIR,
        'final_dataset_file': FINAL_DATASET_FILE,
        'audio_candidates_file': AUDIO_CANDIDATES_FILE,
        'encoder_onnx': ONNX_MODEL_DIR / ENCODER_ONNX_FILENAME,
        'decoder_onnx': ONNX_MODEL_DIR / DECODER_ONNX_FILENAME,
        'joiner_onnx': ONNX_MODEL_DIR / JOINER_ONNX_FILENAME,
        'sample_rate': 16000,
        'n_fft': 400,
        'hop_length': 160,
        'win_length': 400,
        'n_mels': 80,
        # The number of audio frames the model processes at once (T), known from model metadata.
        'chunk_size': 39,
        # The number of frames to shift forward for the next chunk.
        'chunk_shift': 19,
    }

    # --- Setup Logging ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    return config

# ====================================================================================
# Func 1.1: Feature Extraction
# Description: Responsible for extracting Fbank features from an audio file.
# Called by: main -> func_1_5_process_all_audios -> func_1_1_extract_fbank
# ====================================================================================
def func_1_1_extract_fbank(audio_path, config):
    """
    Reads an audio file and extracts Fbank features.
    This function has been simplified to not use per-utterance CMVN.
    """
    try:
        waveform, sr = librosa.load(audio_path, sr=config['sample_rate'])

        # Calculate Mel spectrogram with parameters matching sherpa-onnx.
        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            win_length=config['win_length'],
            n_mels=config['n_mels']
        )

        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Note: Per-utterance CMVN is removed to better align with sherpa-onnx's behavior.
        # A more advanced improvement would be to use a global mean/std.
        features_transposed = log_mel_spectrogram.T

        return features_transposed.astype(np.float32)

    except Exception as e:
        logging.error(f"Failed to process audio file {audio_path}: {e}")
        return None

# ====================================================================================
# Func 1.2: ONNX Model Loader
# Description: Loads the ONNX Encoder model and gets its input/output information.
# Called by: main -> func_1_2_load_onnx_encoder
# ====================================================================================
def func_1_2_load_onnx_encoder(config):
    """
    Loads the ONNX encoder model and extracts its input/output names and shapes.
    """
    # --- Start of modification ---
    # Func 2.1 (sub-function): Helper to map ONNX type strings to numpy dtypes.
    # This is a backward-compatible replacement for ort.get_element_type_name().
    def func_2_1_get_numpy_dtype(onnx_type_str):
        type_map = {
            'tensor(float)': 'float32',
            'tensor(float16)': 'float16',
            'tensor(double)': 'float64',
            'tensor(int8)': 'int8',
            'tensor(int16)': 'int16',
            'tensor(int32)': 'int32',
            'tensor(int64)': 'int64',
            'tensor(uint8)': 'uint8',
            'tensor(uint16)': 'uint16',
            'tensor(uint32)': 'uint32',
            'tensor(uint64)': 'uint64',
            'tensor(bool)': 'bool',
        }
        return type_map.get(onnx_type_str, None)
    # --- End of modification ---

    try:
        logging.info(f"Loading ONNX Encoder model from: {config['encoder_onnx']}")
        session = ort.InferenceSession(str(config['encoder_onnx']))

        # Get names, shapes, and data types for all inputs.
        inputs_info = []
        for i in session.get_inputs():
            # --- Start of modification ---
            # Use our new helper function instead of the missing one.
            numpy_dtype_str = func_2_1_get_numpy_dtype(i.type)
            if numpy_dtype_str is None:
                raise TypeError(f"Unsupported ONNX input type '{i.type}' for input '{i.name}'")
            # --- End of modification ---

            inputs_info.append({'name': i.name, 'shape': i.shape, 'type': numpy_dtype_str})

        # Get names for all outputs.
        outputs_info = [o.name for o in session.get_outputs()]

        logging.info(f"Encoder model loaded successfully. Found {len(inputs_info)} inputs and {len(outputs_info)} outputs.")
        return session, inputs_info, outputs_info

    except Exception as e:
        logging.error(f"Failed to load ONNX encoder model: {e}")
        sys.exit(1)


# ====================================================================================
# Func 1.3: Initial State Generator
# Description: Creates an initial dictionary of zero-filled states for streaming.
# Called by: main -> func_1_5_process_all_audios -> func_1_3_create_initial_states
# ====================================================================================
def func_1_3_create_initial_states(encoder_inputs_info):
    """
    Creates a dictionary of zero-filled numpy arrays for the initial states.
    """
    initial_states = {}
    # Iterate through all inputs, skipping the first one ('x'), to create zero tensors for states.
    for inp_info in encoder_inputs_info[1:]:
        dtype = np.dtype(inp_info['type'])

        # Replace dynamic dimension 'N' with a fixed size of 1.
        shape = [1 if isinstance(dim, str) else dim for dim in inp_info['shape']]
        initial_states[inp_info['name']] = np.zeros(shape, dtype=dtype)

    return initial_states

# ====================================================================================
# Func 1.4: Single Audio Processor
# Description: Simulates streaming processing for one audio file and generates calibration data for each valid step.
# Called by: main -> func_1_5_process_all_audios -> func_1_4_process_single_audio
# ====================================================================================
def func_1_4_process_single_audio(audio_path, encoder_session, encoder_inputs_info, encoder_outputs_info, initial_states, config):
    """
    Simulates streaming inference for a single audio file and generates calibration data points.
    """
    # Func 2.1 (sub-function): Extract features for the entire audio file.
    all_features = func_1_1_extract_fbank(audio_path, config)
    if all_features is None:
        return []

    base_filename = Path(audio_path).stem
    total_frames = all_features.shape[0]
    current_states = initial_states

    generated_lines_for_this_audio = []

    # Simulate a sliding window to process the entire audio.
    for start_frame in range(0, total_frames, config['chunk_shift']):
        end_frame = start_frame + config['chunk_size']
        if end_frame > total_frames:
            break # Discard the last segment if it's smaller than a chunk.

        # Step 1: Prepare the feature chunk for the current window.
        feature_chunk = all_features[start_frame:end_frame, :]
        feature_chunk = np.expand_dims(feature_chunk, axis=0) # Add batch dimension.

        # Step 2: Prepare the complete input dictionary for ONNX Runtime.
        onnx_inputs = {encoder_inputs_info[0]['name']: feature_chunk}
        onnx_inputs.update(current_states)

        # Step 3: Save this complete set of inputs as one line of calibration data.
        current_data_line_paths = []
        step_dir = config['calibration_npy_dir'] / f"{base_filename}_step_{start_frame}"
        step_dir.mkdir(parents=True, exist_ok=True)

        for name, data in onnx_inputs.items():
            npy_path = step_dir / f"{name}.npy"
            np.save(npy_path, data)
            current_data_line_paths.append(str(npy_path.resolve()))

        generated_lines_for_this_audio.append(" ".join(current_data_line_paths))

        # Step 4: Run forward inference with ONNX Runtime to get the next states.
        try:
            outputs = encoder_session.run(encoder_outputs_info, onnx_inputs)

            # Step 5: Update the states for the next iteration.
            # The first 35 outputs are the new states (skip the first 'encoder_out').
            for i, output_name in enumerate(encoder_outputs_info[1:]):
                # The name of the next state input corresponds to the previous state output.
                state_input_name = encoder_inputs_info[i + 1]['name']
                current_states[state_input_name] = outputs[i + 1]

        except Exception as e:
            logging.warning(f"ONNX Runtime inference failed for chunk at {start_frame} in {audio_path}: {e}")
            break # Skip the rest of this audio file if inference fails.

    return generated_lines_for_this_audio

# ====================================================================================
# Func 1.5: Main Processing Loop
# Description: Iterates through all candidate audio files, calls the processor, and collects results.
# Called by: main -> func_1_5_process_all_audios
# ====================================================================================
def func_1_5_process_all_audios(config, encoder_session, encoder_inputs_info, encoder_outputs_info):
    """
    Iterates through all candidate audio files and processes them to generate calibration data.
    """
    try:
        with open(config['audio_candidates_file'], 'r') as f:
            audio_paths = [line.strip() for line in f if line.strip()]
        logging.info(f"Found {len(audio_paths)} audio files to process from {config['audio_candidates_file']}.")
    except FileNotFoundError:
        logging.error(f"Audio candidates file not found: {config['audio_candidates_file']}")
        sys.exit(1)

    # Func 2.2 (sub-function): Create the main directory for .npy files.
    config['calibration_npy_dir'].mkdir(parents=True, exist_ok=True)

    # Func 2.3 (sub-function): Get the initial zero-states.
    initial_states = func_1_3_create_initial_states(encoder_inputs_info)

    all_dataset_lines = []

    # Iterate through and process each audio file.
    for i, audio_path in enumerate(audio_paths):
        logging.info(f"[{i+1}/{len(audio_paths)}] Processing: {audio_path}")
        # Func 2.4 (sub-function): Process a single audio file.
        lines = func_1_4_process_single_audio(
            audio_path,
            encoder_session,
            encoder_inputs_info,
            encoder_outputs_info,
            initial_states,
            config
        )
        all_dataset_lines.extend(lines)

    return all_dataset_lines

# ====================================================================================
# Func 1.6: Final Output Writer
# Description: Writes all collected calibration data lines to the final dataset.txt file.
# Called by: main -> func_1_6_write_final_dataset
# ====================================================================================
def func_1_6_write_final_dataset(lines, config):
    """
    Writes all collected calibration data lines to the final dataset file.
    """
    try:
        with open(config['final_dataset_file'], 'w') as f:
            for line in lines:
                f.write(line + '\n')
        logging.info(f"Successfully generated {len(lines)} lines of calibration data.")
        logging.info(f"Final dataset file saved to: {config['final_dataset_file']}")
    except Exception as e:
        logging.error(f"Failed to write final dataset file: {e}")

# ====================================================================================
# Func 0.0: Main Entry Point
# Description: The main entry point of the script that calls top-level functions in order.
# ====================================================================================
def main():
    """
    Main function to run the entire calibration data generation process.
    """
    # Func 1.0: Load all configurations.
    config = func_1_0_setup_configuration()

    # Func 1.2: Load the ONNX Encoder model.
    encoder_session, encoder_inputs_info, encoder_outputs_info = func_1_2_load_onnx_encoder(config)

    # Func 1.5: Process all audio files to generate calibration data lines.
    # (This will internally call Func 1.1, 1.3, and 1.4)
    all_dataset_lines = func_1_5_process_all_audios(config, encoder_session, encoder_inputs_info, encoder_outputs_info)

    # Func 1.6: Write the results to the final dataset file.
    if all_dataset_lines:
        func_1_6_write_final_dataset(all_dataset_lines, config)
    else:
        logging.warning("No calibration data was generated. The output file will be empty.")


if __name__ == '__main__':
    main()
