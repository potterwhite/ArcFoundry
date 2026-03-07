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
    Description:
        This class replicates the Sherpa-Onnx C++ feature extraction pipeline.
        It ensures that the input data distribution generated in Python matches
        that of the C++ runtime.
    """

    def __init__(self, time_frames=71, sample_rate=16000, n_mels=80):
        # Standard parameters of Sherpa feature extraction
        self.sample_rate = sample_rate
        self.n_fft = 400
        self.hop_length = 160
        self.win_length = 400
        self.n_mels = n_mels

        # Key preprocessing parameters (sourced from old-files/09_...py)
        self.dither = 0.0
        self.remove_dc_offset = True
        self.preemph_coeff = 0.97
        self.log_zero_guard = 1e-10

        self.frames = time_frames

    def compute(self, audio_path: str) -> np.ndarray:
        """
        Description:
            This method processes the input audio file and computes the Log-Mel
            Spectrogram, following the Sherpa-Onnx feature extraction steps.
        """
        try:
            # 1. Load Audio
            #    Resampling
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate)

            # 2. Dithering
            if self.dither > 0.0:
                waveform += self.dither * np.random.randn(*waveform.shape)

            # 3. Remove DC Offset
            if self.remove_dc_offset:
                waveform -= np.mean(waveform)

            # 4. Pre-emphasis
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
            #    Note: Using log(max(x, 1e-10)) instead of power_to_db to align with C++
            log_mel = np.log(np.maximum(mel, self.log_zero_guard))

            # 7. Transpose to [Time, Feature]
            feats = log_mel.T.astype(np.float32)

            # 8. Pad to target frames
            T, F = feats.shape
            logger.info(f"[DSP] Features shape: {T}, {F}")

            if T > self.frames:
                feats = feats[:self.frames, :]
                logger.info(f"[DSP] Features shape contruncated after padding: {feats.shape}")
            elif T < self.frames:
                pad = np.zeros((self.frames - T, F), dtype=feats.dtype)
                feats = np.concatenate([feats, pad], axis=0)
                logger.info(f"[DSP] Features shape padded after padding: {feats.shape}")

            return feats

        except Exception as e:
            logging.error(f"[DSP] Feature extraction failed for {audio_path}: {e}")
            raise e
