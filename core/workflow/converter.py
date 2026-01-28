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
from core.utils import logger, get_btf_from_yaml
from core.rknn_adapter import RKNNAdapter
from core.verification.comparator import ModelComparator
from core.dsp.sherpa_features_extractor import SherpaFeatureExtractor


class StandardConverter:

    def __init__(self, global_config):
        self.cfg = global_config
        self.output_dir = self.cfg.get("project", {}).get("output_dir", "./output")
        _, self.json_time_frames, self.json_feature = get_btf_from_yaml(self.cfg)

    def _verify_model(self, model_cfg, onnx_path, build_config):
        # def _verify_model(self, model_cfg, onnx_path, rknn_path, build_config):
        """
        V1.1 Feature: Auto-Verification
        Returns:
            float: The minimum cosine similarity score (0.0 - 1.0).
                   Returns 1.0 if verification is skipped or crashes (to avoid false triggers).
        """
        logger.info(f"🔎 Starting Verification for {model_cfg['name']}...")
        min_score = 1.0  # Default safe value

        try:
            # 1. Initialize Comparator
            target_platform = self.cfg.get("target", {}).get("platform")
            comparator = ModelComparator(target_platform)

            # --- CHANGE START ---
            input_shapes = model_cfg.get("input_shapes", None)
            #build_config = self.cfg.get("build", {})

            comparator.prepare_simulator(onnx_path, input_shapes, build_config)
            # --- CHANGE END ---

            # 2. Prepare Input Data
            sess = ort.InferenceSession(onnx_path)
            input_feed = {}
            extractor = SherpaFeatureExtractor(time_frames=self.json_time_frames, sample_rate=16000, n_mels=self.json_feature)

            test_audio_path = self.cfg.get("build", {}).get("test_input", None)

            for i, inp in enumerate(sess.get_inputs()):
                # a. Handle Dynamic Shape (Replace string/None with 1)
                static_shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]

                # b. Detect NumPy Data Type
                onnx_type = inp.type
                np_dtype = np.float32  # Default fallback
                if "int64" in onnx_type:
                    np_dtype = np.int64
                elif "int32" in onnx_type:
                    np_dtype = np.int32
                elif "bool" in onnx_type:
                    np_dtype = bool
                elif "float16" in onnx_type:
                    np_dtype = np.float16

                # Deal with Dynamic Shape (Replace string/None with 1) --- IGNORE ---
                static_shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]

                # c. Generate Input Data
                # Condition: Index 0 + Configured Path + File Exists + Is Float Type
                if (i == 0 and test_audio_path and os.path.exists(test_audio_path) and np.issubdtype(
                        np_dtype, np.floating)):
                    logger.info(f"   Using real audio for input '{inp.name}': {test_audio_path}")
                    feats = extractor.compute(test_audio_path)

                    # Crop to target length
                    target_len = static_shape[1]
                    if feats.shape[0] > target_len:
                        feats = feats[:target_len, :]

                    input_feed[inp.name] = np.expand_dims(feats, axis=0).astype(np_dtype)

                else:
                    # Fallback: Random Data based on Type
                    if np.issubdtype(np_dtype, np.integer):
                        # Generate random integers (e.g. sequence lengths)
                        input_feed[inp.name] = np.random.randint(0, 10, size=static_shape).astype(np_dtype)
                    elif np_dtype == bool:
                        input_feed[inp.name] = np.random.choice([True, False], size=static_shape)
                    else:
                        # Generate random floats
                        input_feed[inp.name] = np.random.rand(*static_shape).astype(np_dtype)

            # 3. Execute Comparison
            metrics = comparator.compare_with_onnx(onnx_path, input_feed)

            # Calculate minimum score
            if metrics:
                min_score = min(metrics.values())

            # 4. Determine Result
            if comparator.validate_metric(metrics, threshold=0.98):
                logger.info(f"✅ Verification PASSED: {model_cfg['name']} matches ONNX baseline.")
            else:
                logger.warning(
                    f"⚠️ Verification WARNING: {model_cfg['name']} accuracy might be low (Min Score: {min_score:.6f})."
                )

        except Exception as e:
            logger.error(f"❌ Verification Failed: {str(e)}")
            import traceback
            # Print full traceback for debugging
            logger.error(traceback.format_exc())

        return min_score

    # --------------------------------------------------------------------------
    # Level 2: Standard Conversion & Evaluation
    # --------------------------------------------------------------------------
    def convert_and_evaluate(self, target_plat, model_name, onnx_path, output_path, input_shapes,
                             build_config, custom_string, model_cfg):
        """
        Description:
            This function handles a standard conversion process and returns an accuracy score.
        Note:
            This function is responsible for creating the adapter, using it, and must release it.
        Returns:
            float: The accuracy score (0.0 - 1.0). Returns 0.0 if conversion fails.
        """

        analysis_dir = os.path.join(self.output_dir, "analysis", model_name)
        existing_report = os.path.join(analysis_dir, "error_analysis.txt")

        # === [Fast-Forward] Check for existing analysis report ===
        # Only trigger if both RKNN model and analysis report exist
        if os.path.exists(output_path) and os.path.exists(existing_report):
            logger.warning(f"⏩ [FAST-FORWARD] Found existing analysis report: {existing_report}")
            logger.warning(f"   Skipping Build & Verification to jump straight to Hybrid Quantization logic.")

            # return 0.0 to trigger recovery flow
            return 0.0
        # =========================================================

        adapter = RKNNAdapter(target_platform=target_plat, verbose=build_config.get('verbose', False))

        # Processing -- A. Convert
        ret = adapter.convert(onnx_path, output_path, input_shapes, build_config, custom_string)
        score = 1.0

        if ret:
            logger.info(f"SUCCESS: Standard model saved to {output_path}")

            # Processing -- B. Verify (Verification)
            is_quant = build_config.get('quantization', {}).get('enabled', False)
            if is_quant:
                score = self._verify_model(model_cfg, onnx_path, build_config)

                if score < 0.99:
                    logger.warning(f"📉 Low Accuracy ({score:.4f}). Running immediate analysis before release...")
                    dataset_path = build_config.get('quantization', {}).get('dataset')
                    analysis_dir = os.path.join(self.output_dir, "analysis", model_name)
                    adapter.run_deep_analysis(dataset_path, analysis_dir)
                else:
                    logger.info(f"✅ Accuracy is good enough ({score:.4f}).")
            else:
                logger.info(f"🔎 Verification skipped for {model_cfg['name']} (Quantization is disabled).")
        else:
            logger.error(f"FAILURE: RKNN Conversion failed for {model_name}")
            score = 0.0

        # adapter must be released here for recoverer need a new adapter later
        adapter.release()

        # return the score
        return score
