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

import numpy as np
import librosa
import logging


class SherpaFeatureExtractor:
    """
    专门用于复刻 Sherpa-Onnx C++ 底层特征提取流程的 DSP 模块。
    确保 Python 端生成的输入数据分布与 C++ Runtime 完全一致。
    """

    def __init__(self, sample_rate=16000, n_mels=80):
        # Sherpa Zipformer 的标准参数
        self.sample_rate = sample_rate
        self.n_fft = 400
        self.hop_length = 160
        self.win_length = 400
        self.n_mels = n_mels

        # 关键预处理参数 (来源于 old-files/09_...py)
        self.dither = 0.0
        self.remove_dc_offset = True
        self.preemph_coeff = 0.97
        self.log_zero_guard = 1e-10

    def compute(self, audio_path: str) -> np.ndarray:
        """
        读取音频并计算 Log-Mel Spectrogram (Shape: [N, 80])
        """
        try:
            # 1. Load Audio
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate)

            # 2. Dithering (防死锁/增加鲁棒性，虽默认为0但保留接口)
            if self.dither > 0.0:
                waveform += self.dither * np.random.randn(*waveform.shape)

            # 3. Remove DC Offset (去直流) - 关键步骤
            if self.remove_dc_offset:
                waveform -= np.mean(waveform)

            # 4. Pre-emphasis (预加重) - 关键步骤
            if self.preemph_coeff > 0.0:
                waveform = librosa.effects.preemphasis(waveform, coef=self.preemph_coeff)

            # 5. Mel Spectrogram
            mel = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
            )

            # 6. Log (Power -> DB)
            # 注意：这里使用 log(max(x, 1e-10)) 而不是 power_to_db，以对齐 C++
            log_mel = np.log(np.maximum(mel, self.log_zero_guard))

            # 7. Transpose to [Time, Feature]
            return log_mel.T.astype(np.float32)

        except Exception as e:
            logging.error(f"[DSP] Feature extraction failed for {audio_path}: {e}")
            raise e
