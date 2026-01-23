# Copyright (c) 2026 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# core/quantization/strategies/streaming.py

import os
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
from core.utils import logger, ensure_dir
from core.dsp.sherpa_features_extractor import SherpaFeatureExtractor
from . import register_strategy


@register_strategy("streaming_audio")
class StreamingAudioStrategy:
    """
    Calibration strategy for streaming audio models (e.g., Sherpa-Zipformer/Transducer).

    This strategy handles:
    1. Audio feature extraction (DSP).
    2. ONNX Runtime state management (Simulating streaming inference).
    3. Sliding window slicing.
    4. Saving intermediate tensors (.npy) for quantization calibration.
    """

    def __init__(self, config):
        self.cfg = config
        # Default sampling interval: save data every 5 frames
        self.sampling_interval = self.cfg.get('build', {}).get('quantization', {}).get('sampling_interval', 5)
        self.sherpa_extractor = SherpaFeatureExtractor()

    def _load_audio_list(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        with open(dataset_path, 'r') as f:
            return [x.strip() for x in f.readlines() if x.strip()]

    def _get_numpy_dtype(self, onnx_type):
        mapping = {
            'tensor(float)': np.float32,
            'tensor(float16)': np.float16,
            'tensor(int64)': np.int64,
            'tensor(int32)': np.int32,
            'tensor(bool)': bool
        }
        return mapping.get(onnx_type, np.float32)

    def run(self, onnx_path, output_dir, dataset_path):
        """
        Executes the calibration generation process.

        Args:
            onnx_path (str): Path to the source ONNX model.
            output_dir (str): Workspace directory to save .npy files and the list.
            dataset_path (str): Path to the text file containing audio file paths.

        Returns:
            str: Path to the generated dataset_list.txt.
        """
        # Preparation -- 1. Load audio paths
        audio_paths = self._load_audio_list(dataset_path)
        logger.info(f"⚡ [Strategy: Streaming] Generating calibration data from {len(audio_paths)} files...")

        # Preparation -- 2. Setup directories
        npy_dir = os.path.join(output_dir, "calibration_data")
        ensure_dir(npy_dir)

        # Preparation -- 3. Get ONNX model I/O metadata
        # The inputs and outputs of the entire graph are fixed.
        # There are exactly 36 inputs (including x) and 36 outputs (including encoder_out).
        # These are scattered throughout the graph.
        # so the "ort.InferenceSession.get_inputs and get_outputs" are essentially retrieving the information for these 72 nodes in the NodeArg format
        sess = ort.InferenceSession(onnx_path)
        inputs_meta = sess.get_inputs()
        outputs_meta = sess.get_outputs()

        # Preparation -- 4. Initialize states with zeros (Assumption: input[0] is feature, input[1:] are states)
        initial_states = {}
        for inp in inputs_meta[1:]:
            dtype = self._get_numpy_dtype(inp.type)
            # ******
            # Handle dynamic shapes: replace string/'None' with 1
            # shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]
            # ------
            shape = []  # New a list to hold processed shape

            for d in inp.shape:
                # Check if the dimension is a string (e.g., "batch_size") or None
                if isinstance(d, str) or d is None:
                    shape.append(1)  # If it's a dynamic dimension, force it to 1
                else:
                    shape.append(d)  # If it's a normal number (fixed dimension), keep it as is
            # ******
            initial_states[inp.name] = np.zeros(shape, dtype=dtype)

        # Preparation -- 5. Define variables
        generated_lines = []
        CHUNK_SIZE = 39  # Fixed chunk size for Sherpa
        CHUNK_SHIFT = 19  # Fixed stride for Sherpa

        # Processing -- 1. Main Loop: Process each audio file
        for audio_path in tqdm(audio_paths, desc="Calibrating (Streaming)"):
            try:
                # DSP Feature Extraction [T, 80]
                # Processing -- a. Extract features with sherpa`s model settings
                all_features = self.sherpa_extractor.compute(audio_path)
                total_frames = all_features.shape[0]

                current_states = initial_states.copy()
                step_counter = 0

                # Processing -- b. Sliding Window Inference Simulation
                for start in range(0, total_frames, CHUNK_SHIFT):
                    end = start + CHUNK_SIZE
                    if end > total_frames:
                        break

                    step_counter += 1

                    # Prepare Input Feed
                    feature_chunk = np.expand_dims(all_features[start:end, :], axis=0)  # [1, 39, 80]
                    feed_dict = {inputs_meta[0].name: feature_chunk}
                    feed_dict.update(current_states)

                    # [Sampling] Save .npy files if interval is met
                    if (step_counter - 1) % self.sampling_interval == 0:
                        step_files = []
                        step_name = f"{Path(audio_path).stem}_s{start}"

                        for name, data in feed_dict.items():
                            safe_name = name.replace('/', '_').replace(':', '_')
                            file_path = os.path.join(npy_dir, f"{step_name}_{safe_name}.npy")
                            np.save(file_path, data)
                            step_files.append(os.path.abspath(file_path))

                        generated_lines.append(" ".join(step_files))

                    # [State Update] Run Inference to get next states
                    outputs = sess.run(None, feed_dict)

                    # Map outputs back to inputs for next step
                    # Assumption: Output[i+1] corresponds to Input[i+1]
                    new_states = {}
                    for i, _ in enumerate(outputs_meta[1:]):
                        input_name_for_state = inputs_meta[i + 1].name
                        new_states[input_name_for_state] = outputs[i + 1]

                    current_states = new_states

            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                continue

        # 5. Write List File
        list_file_path = os.path.join(output_dir, "dataset_list.txt")
        with open(list_file_path, "w") as f:
            f.write("\n".join(generated_lines))

        logger.info(f"Calibration dataset ready: {len(generated_lines)} samples.")
        return list_file_path
