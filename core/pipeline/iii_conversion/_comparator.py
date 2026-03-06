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
import onnxruntime as ort
from rknn.api import RKNN
from core.utils.utils import logger
from typing import List, Dict, Any


class ModelComparator:
    """
    Responsibilities:
        - Compare the precision between ONNX and RKNN Simulator.
    Steps:
        - Start Simulator Mode via Shadow Build
        - Run Dual Inference (ONNX & RKNN Simulator)
        - Compute Similarity Metrics (Cosine Similarity)
    """

    def __init__(self, target_platform: str):
        self.target_platform = target_platform
        self.rknn = RKNN(verbose=False)

    @staticmethod
    def validate_metric(metrics: Dict[str, float], threshold=0.98) -> bool:
        all_pass = True
        for name, score in metrics.items():
            if score < threshold:
                all_pass = False
        return all_pass

    def prepare_simulator(self, onnx_path: str, input_shapes: List[List[int]], build_config: Dict[str, Any]):
        """
        Description:
            Start simulator in memory via compiling the model.
            This method is totally calls RKNN APIs.
        """
        # Preparation -- 1. Echo welcome message
        logger.info(f"[Verify] Initializing Simulator Environment for {self.target_platform}...")

        # Preparation -- 2. Configure RKNN object
        json_quant_enable = build_config.get('quantization', {}).get('enabled', False)
        json_ds_path = build_config.get('quantization', {}).get('dataset', None)

        # Processing -- 1. Configure platform
        self.rknn.config(target_platform=self.target_platform)

        # Processing -- 2. Load ONNX model
        if self.rknn.load_onnx(model=onnx_path, input_size_list=input_shapes) != 0:
            raise RuntimeError("Simulator: Load ONNX failed!")

        # Processing -- 3. Build (Shadow Build)
        # Note: We inherit the build config from the main flow (e.g., quantization params) to ensure consistency.
        #       However, we force do_quantization=False to verify FP16 baseline, or decide based on build_config.
        ret = self.rknn.build(json_quant_enable, json_ds_path)
        if ret != 0:
            raise RuntimeError("Simulator: Build failed!")

        # Processing -- 4. Init Runtime
        #    (target=None triggers simulator mode)
        if self.rknn.init_runtime(target=None) != 0:
            raise RuntimeError("Simulator: Init Runtime failed! (Is rknn-toolkit2 installed correctly?)")

        # Processing -- 5. Finalization
        logger.info("[Verify] Simulator Ready.")

    # def compare_with_onnx(self, onnx_path: str, inputs: Dict[str, np.ndarray]) -> Dict[str, float]:
    #     """
    #     Description:
    #         Dual Inference & Similarity Computation
    #     Args:
    #         onnx_path (str): Path to the ONNX model.
    #         inputs (Dict[str, np.ndarray]): Input data for inference.
    #     Returns:
    #         Dict[str, float]: Similarity metrics for each output.
    #     """

    #     # 1. ONNX Inference
    #     logger.info("[Verify] Running Baseline ONNX Inference...")
    #     sess = ort.InferenceSession(onnx_path)
    #     onnx_input_names = [i.name for i in sess.get_inputs()]
    #     onnx_feed = {name: inputs[name] for name in onnx_input_names if name in inputs}
    #     onnx_outputs = sess.run(None, onnx_feed)

    #     # 2. RKNN Inference
    #     rknn_feed_list = [inputs[name] for name in onnx_input_names]

    #     logger.info("[Verify] Running RKNN Simulator Inference...")
    #     rknn_outputs = self.rknn.inference(inputs=rknn_feed_list, data_format='nchw')

    #     # 3. Compute Metrics
    #     metrics = {}
    #     onnx_output_names = [o.name for o in sess.get_outputs()]

    #     min_len = min(len(onnx_outputs), len(rknn_outputs))

    #     for idx in range(min_len):
    #         name = onnx_output_names[idx]
    #         out_onnx = onnx_outputs[idx].flatten()
    #         out_rknn = rknn_outputs[idx].flatten()

    #         # Handle NaN/Inf cases
    #         if not np.isfinite(out_onnx).all() or not np.isfinite(out_rknn).all():
    #             logger.warning(f"[Verify] Output '{name}' contains NaN/Inf!")
    #             metrics[name] = 0.0
    #             continue

    #         # Cosine Similarity
    #         dot_product = np.dot(out_onnx, out_rknn)
    #         norm_a = np.linalg.norm(out_onnx)
    #         norm_b = np.linalg.norm(out_rknn)

    #         if norm_a == 0 or norm_b == 0:
    #             cos_sim = 1.0 if norm_a == norm_b else 0.0
    #         else:
    #             cos_sim = dot_product / (norm_a * norm_b)

    #         metrics[name] = cos_sim
    #         logger.info(f"   📊 Metric: Output '{name}' Cosine Similarity = {cos_sim:.6f}")

    #     self.rknn.release()
    #     return metrics
    def compare_with_onnx(self, onnx_path: str, inputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Description:
            Performs a dual-inference validation to compare the accuracy of the quantized model (RKNN)
            against the baseline floating-point model (ONNX).

        Args:
            onnx_path (str): The file path to the reference ONNX model.
            inputs (Dict[str, np.ndarray]): A dictionary containing input data, keyed by input name.

        Returns:
            Dict[str, float]: A dictionary mapping output layer names to their Cosine Similarity scores.
                              Score range: [-1.0, 1.0], where 1.0 means identical.
        """
        logger.info(f"🔎 [Verify] Starting comparison logic...")

        # =======================================================
        # Step 1: Baseline Inference (ONNX Runtime)
        # =======================================================
        # We run the original ONNX model (FP32) to get the "Ground Truth".
        try:
            logger.info("[Verify] Running Baseline ONNX Inference...")
            sess = ort.InferenceSession(onnx_path)

            # 1.1 Get input metadata from the ONNX model
            # We need to know exactly what input names the model expects.
            onnx_inputs_meta = sess.get_inputs()
            onnx_input_names = []

            # Explicit loop to extract input names
            for inp in onnx_inputs_meta:
                onnx_input_names.append(inp.name)

            # 1.2 Construct the feed dictionary
            # Match the provided 'inputs' data with the model's required input names.
            onnx_feed = {}
            for name in onnx_input_names:
                if name in inputs:
                    onnx_feed[name] = inputs[name]
                else:
                    # Warn if the user forgot to provide a required input
                    logger.warning(f"⚠️ Input '{name}' required by ONNX but missing in provided inputs.")

            # 1.3 Run Inference
            # 'None' means we fetch all output nodes defined in the model.
            baseline_outputs = sess.run(None, onnx_feed)

            # 1.4 Get output names
            # We need these names to map the results back to readable keys (e.g., 'output_1').
            onnx_outputs_meta = sess.get_outputs()
            onnx_output_names = []
            for out in onnx_outputs_meta:
                onnx_output_names.append(out.name)

        except Exception as e:
            logger.error(f"❌ ONNX Inference failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

        # =======================================================
        # Step 2: Target Inference (RKNN Simulator)
        # =======================================================
        # We run the quantized RKNN model using the PC Simulator.
        try:
            logger.info("[Verify] Running RKNN Simulator Inference...")

            # 2.1 Prepare input list strictly following ONNX input order.
            # Unlike ONNX Runtime (which uses a Dict), RKNN interface typically requires
            # a List of arrays. The order MUST match the ONNX input definition order.
            rknn_feed_list = []
            for name in onnx_input_names:
                # We assume inputs[name] exists because we checked it in Step 1.2
                data_array = inputs[name]
                rknn_feed_list.append(data_array)

            # 2.2 Run Inference
            # data_format='nchw' is the standard layout for models converted from ONNX/PyTorch.
            target_outputs = self.rknn.inference(inputs=rknn_feed_list, data_format='nchw')

        except Exception as e:
            logger.error(f"❌ RKNN Inference failed: {e}")
            self.rknn.release()  # Release resources to avoid memory leaks
            return {}

        # =======================================================
        # Step 3: Compute Similarity Metrics (The Core Algorithm)
        # =======================================================
        # We compare the Baseline (FP32) vs Target (Int8) output by output.
        metrics = {}

        # Determine the number of outputs to compare.
        # Usually they match, but we use min() to prevent index out of bounds just in case.
        num_outputs = min(len(baseline_outputs), len(target_outputs))

        for idx in range(num_outputs):
            output_name = onnx_output_names[idx]

            # 3.1 Flatten arrays to 1D vectors
            # Output shapes might differ slightly (e.g., (1, 100) vs (100,)).
            # Flattening ensures we strictly compare the data sequence, ignoring dimensions.
            vec_baseline = baseline_outputs[idx].flatten()
            vec_target = target_outputs[idx].flatten()

            # 3.2 Safety Check: NaN (Not a Number) or Inf (Infinity)
            # If the model produced garbage (NaN/Inf), calculation is meaningless.
            if not np.isfinite(vec_baseline).all() or not np.isfinite(vec_target).all():
                logger.warning(f"⚠️ [Verify] Output '{output_name}' contains NaN or Inf values!")
                metrics[output_name] = 0.0
                continue

            # 3.3 Calculate Cosine Similarity
            # ---------------------------------------------------------
            # Why Cosine Similarity?
            #
            # 1. Principle: It measures the cosine of the angle between two non-zero vectors.
            #    Formula: Similarity = (A . B) / (||A|| * ||B||)
            #
            # 2. Feature Direction vs. Magnitude:
            #    In Neural Networks, the "direction" of the feature vector usually represents
            #    the pattern or category, while the "magnitude" represents intensity.
            #    Quantization often affects magnitude (due to scaling factors) but preserves direction.
            #
            # 3. Robustness:
            #    Euclidean distance (MSE) is very sensitive to absolute values. A slight global shift
            #    in scale would result in a huge error, even if the model's logic is correct.
            #    Cosine Similarity focuses on the "shape" of the data, making it the standard
            #    metric for verification.
            # ---------------------------------------------------------

            # A. Calculate Dot Product (The numerator)
            dot_product = np.dot(vec_baseline, vec_target)

            # B. Calculate Norm (Magnitude) (The denominator components)
            norm_baseline = np.linalg.norm(vec_baseline)
            norm_target = np.linalg.norm(vec_target)

            # C. Calculate Result with Zero-Division Protection
            if norm_baseline == 0 or norm_target == 0:
                # Edge Case: One of the vectors is all zeros.
                # If both are zero, they match (1.0). If only one is zero, they differ (0.0).
                cos_sim = 1.0 if norm_baseline == norm_target else 0.0
            else:
                cos_sim = dot_product / (norm_baseline * norm_target)

            # 3.4 Store and Log the result
            metrics[output_name] = cos_sim

            # Add visual indicators for log readability
            if cos_sim < 0.90:
                status_icon = "❌"  # Critical failure
            elif cos_sim < 0.98:
                status_icon = "⚠️"  # Warning
            else:
                status_icon = "✅"  # Pass

            logger.info(f"   📊 Metric: {status_icon} Output '{output_name}' Cosine Similarity = {cos_sim:.6f}")

        # =======================================================
        # Step 4: Cleanup
        # =======================================================
        # Release the virtual NPU resources to free up memory
        self.rknn.release()

        return metrics
