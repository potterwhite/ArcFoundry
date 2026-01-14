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

import os
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
from core.utils import logger, ensure_dir
from core.dsp.audio_features import SherpaFeatureExtractor


class CalibrationGenerator:
    """
    负责生成 RKNN 量化所需的校准数据集 (Calibration Dataset)。

    对于流式模型 (如 Sherpa-Zipformer)，它不仅提取音频特征，
    还会运行 ONNX Runtime 进行预推理，以捕获真实的 State (Hidden States) 分布。
    """

    def __init__(self, config):
        self.cfg = config
        # 从配置中读取采样间隔 (防止数据量爆炸)
        # 默认为 5，即每 5 帧保存一次数据
        self.sampling_interval = self.cfg.get('build', {}).get(
            'quantization', {}).get('sampling_interval', 5)

        self.dsp = SherpaFeatureExtractor()

    def _load_audio_list(self, dataset_path):
        """读取 dataset 配置指定的音频列表文件"""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Calibration dataset file not found: {dataset_path}")

        with open(dataset_path, 'r') as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        return lines

    def _get_numpy_dtype(self, onnx_type):
        """Map ONNX tensor type to Numpy dtype"""
        mapping = {
            'tensor(float)': np.float32,
            'tensor(float16)': np.float16,
            'tensor(int64)': np.int64,
            'tensor(int32)': np.int32,
            'tensor(bool)': bool
        }
        return mapping.get(onnx_type, np.float32)

    def generate(self, onnx_path, output_dir):
        """
        主入口：生成校准数据

        Args:
            onnx_path: 中间态 ONNX 模型路径 (用于获取 State Shape)
            output_dir: .npy 文件和列表文件的保存目录

        Returns:
            dataset_list_path: 生成的 .txt 文件路径 (传给 RKNN build 使用)
        """
        # 1. 获取数据集路径
        dataset_cfg = self.cfg.get('build', {}).get('quantization',
                                                    {}).get('dataset', None)
        if not dataset_cfg:
            logger.warning(
                "Quantization is enabled but no 'dataset' path provided in config."
            )
            return None

        audio_paths = self._load_audio_list(dataset_cfg)
        logger.info(
            f"⚡ Generating calibration data from {len(audio_paths)} audio files..."
        )
        logger.info(f"   Sampling Interval: {self.sampling_interval}")

        # 2. 初始化环境
        npy_dir = os.path.join(output_dir, "calibration_data")
        list_file_path = os.path.join(output_dir, "dataset_list.txt")

        # === [V1.1 Part 2 Improved] Smart Caching ===
        # 如果 List 文件存在，且里面的第一行对应的文件也存在，我们认为缓存有效
        if os.path.exists(list_file_path):
            with open(list_file_path, 'r') as f:
                first_line = f.readline().strip()

            # 检查第一行的第一个文件是否存在 (简单的冒烟检查)
            if first_line:
                first_npy = first_line.split(' ')[0]
                if os.path.exists(first_npy):
                    logger.info(
                        f"⚡ [Cache Hit] Found existing calibration data at: {list_file_path}"
                    )
                    logger.info(
                        f"   Skipping regeneration. (Delete {list_file_path} to force regenerate)"
                    )
                    return list_file_path
        # ============================================

        ensure_dir(npy_dir)
        audio_paths = self._load_audio_list(dataset_cfg)

        sess = ort.InferenceSession(onnx_path)
        inputs_meta = sess.get_inputs()
        outputs_meta = sess.get_outputs()

        # 3. 初始化 States (全零)
        # 假设: input[0] 是 features, input[1:] 是 states
        # 这是一个针对 Sherpa 的强假设，未来可能需要更通用的逻辑
        initial_states = {}
        for inp in inputs_meta[1:]:
            dtype = self._get_numpy_dtype(inp.type)
            # 处理动态 Shape: 将字符串或None替换为1
            shape = [
                1 if isinstance(d, str) or d is None else d for d in inp.shape
            ]
            initial_states[inp.name] = np.zeros(shape, dtype=dtype)

        # 4. 遍历音频并处理
        generated_lines = []

        # Sherpa 参数: 39 frames per chunk, shift 19 (Should match model)
        # 理想情况下这些应该从 Metadata 读取，现在先 Hardcode (和 old scripts 一致)
        CHUNK_SIZE = 39
        CHUNK_SHIFT = 19

        for audio_path in tqdm(audio_paths, desc="Calibrating"):
            try:
                # DSP Feature Extraction
                all_features = self.dsp.compute(audio_path)  # [T, 80]

                total_frames = all_features.shape[0]
                current_states = initial_states.copy()
                step_counter = 0

                # Sliding Window
                for start in range(0, total_frames, CHUNK_SHIFT):
                    end = start + CHUNK_SIZE
                    if end > total_frames:
                        break  # Drop last partial chunk

                    step_counter += 1

                    # 准备当前帧的输入数据
                    feature_chunk = np.expand_dims(all_features[start:end, :],
                                                   axis=0)  # [1, 39, 80]

                    feed_dict = {inputs_meta[0].name: feature_chunk}
                    feed_dict.update(current_states)

                    # --- [Sampling Strategy] ---
                    # 只有满足间隔才保存 .npy，但推理必须每一步都跑以更新 State
                    if (step_counter - 1) % self.sampling_interval == 0:
                        step_files = []
                        step_name = f"{Path(audio_path).stem}_s{start}"

                        # Save each input tensor as .npy
                        for name, data in feed_dict.items():
                            safe_name = name.replace('/',
                                                     '_').replace(':', '_')
                            file_path = os.path.join(
                                npy_dir, f"{step_name}_{safe_name}.npy")
                            np.save(file_path, data)
                            step_files.append(os.path.abspath(file_path))

                        # Add to list (Space separated)
                        generated_lines.append(" ".join(step_files))

                    # --- [Run Inference to Update States] ---
                    # 必须跑一次推理才能拿到下一帧正确的 State
                    # 注意：我们不需要拿到所有输出，只需要 State 部分
                    outputs = sess.run(None, feed_dict)

                    # 更新 States
                    # 假设 Output 顺序: [out, state1, state2...] 对应 Input [in, state1, state2...]
                    # Sherpa 输出的第一个通常是 encoder_out，后面是 next_states
                    new_states = {}
                    for i, out_meta in enumerate(outputs_meta[1:]):
                        # 输出的第 i+1 个对应输入的第 i+1 个 (错开第一个)
                        input_name_for_state = inputs_meta[i + 1].name
                        new_states[input_name_for_state] = outputs[i + 1]

                    current_states = new_states

            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                continue

        # 5. 写入最终列表文件
        list_file_path = os.path.join(output_dir, "dataset_list.txt")
        with open(list_file_path, "w") as f:
            f.write("\n".join(generated_lines))

        logger.info(
            f"Calibration dataset ready: {len(generated_lines)} samples.")
        return list_file_path
