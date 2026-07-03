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

# core/quantization/strategies/vision.py

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import logger, ensure_dir
from . import register_strategy


@register_strategy("vision")
class VisionQuantizationStrategy:
    """
    Calibration strategy for computer vision models (e.g., MODNet, YOLO, ResNet).

    This strategy handles:
    1. Image loading and preprocessing
    2. Resizing to model input dimensions
    3. Normalization (if specified)
    4. Saving preprocessed images as .npy files for quantization calibration
    """

    def __init__(self, config):
        self.cfg = config
        # Default sampling interval: save every 5th image
        self.sampling_interval = self.cfg.get('build', {}).get(
            'quantization', {}).get('sampling_interval', 5)

        # Extract normalization parameters from config
        self.mean_values = None
        self.std_values = None

        # Try to get normalization from models config
        models = self.cfg.get('models', [])
        if models:
            model_config = models[0]  # Use first model's config
            normalization = model_config.get('normalization', {})
            if normalization:
                self.mean_values = normalization.get(
                    'mean_values', [[127.5, 127.5, 127.5]])[0]
                self.std_values = normalization.get('std_values',
                                                    [[127.5, 127.5, 127.5]])[0]

    def _load_image_list(self, dataset_path):
        """Load list of image paths from dataset file."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        with open(dataset_path, 'r') as f:
            return [x.strip() for x in f.readlines() if x.strip()]

    def _preprocess_image(self, image_path, target_shape):
        """
        Preprocess a single image for calibration.

        Args:
            image_path: Path to the image file
            target_shape: Target shape (H, W, C) for the model input

        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target dimensions
        target_h, target_w = target_shape[0], target_shape[1]
        img = cv2.resize(img, (target_w, target_h),
                         interpolation=cv2.INTER_AREA)

        # Convert to float32
        img = img.astype(np.float32)

        # Apply normalization if specified
        # Note: RKNN will handle normalization internally if mean/std are configured
        # For calibration, we typically provide unnormalized data (0-255 range)
        # The normalization parameters are used during model conversion, not here

        return img

    def _is_multi_input(self, input_shapes):
        """
        Detect whether the model requires more than one input tensor.

        WHY THIS EXISTS:
        ----------------
        Traditional CV models (MODNet, YOLO, ResNet ...) have a single input:
        one image tensor.  But recurrent video models like RVM (Robust Video
        Matting) have *multiple* inputs:

            RVM inputs (5 total):
              src  [1,3,256,256]   -- the RGB image frame
              r1i  [1,16,32,32]   -- ConvGRU hidden state level-1
              r2i  [1,20,16,16]   -- ConvGRU hidden state level-2
              r3i  [1,40,8,8]     -- ConvGRU hidden state level-3
              r4i  [1,64,4,4]     -- ConvGRU hidden state level-4

        RKNN Toolkit 2's rknn.build(dataset=...) requires the calibration list
        to provide data for *every* input of the model.  If the list only has
        paths for 1 input but the model expects 5, you get:

            ValueError: The input num: 1 ... not match input num: 5 in model!

        So we need to know early whether we are dealing with a multi-input
        model so we can generate the correct calibration file format.

        HOW WE DETECT IT:
        -----------------
        In the YAML config, `input_shapes` comes in two flavours:

          - Old list format:  [[1,3,256,256]]          --> single input
          - New dict format:  {src: [1,3,256,256],      --> could be 1 or N
                               r1i: [1,16,32,32], ...}

        Only the dict format carries input *names*, which is necessary for
        multi-input.  If it is a dict with >1 key, the model is multi-input.
        """
        if isinstance(input_shapes, dict):
            return len(input_shapes) > 1
        return False

    def run(self, onnx_path, output_dir, dataset_path):
        """
        Generate calibration data (.npy files + list) for RKNN INT8 quantization.

        Supports two scenarios:

        1. SINGLE-INPUT models (MODNet, YOLO, ResNet ...):
           - One .npy file per calibration sample (the preprocessed image).
           - calibration_list.txt has one absolute path per line:
                 /abs/path/calib_0000.npy
                 /abs/path/calib_0005.npy

        2. MULTI-INPUT models (RVM with ConvGRU recurrent states):
           - N .npy files per sample (1 image + N-1 auxiliary tensors).
           - calibration_list.txt has N space-separated paths per line:
                 /abs/calib_0000_src.npy /abs/calib_0000_r1i.npy ... /abs/calib_0000_r4i.npy
                 /abs/calib_0005_src.npy /abs/calib_0005_r1i.npy ... /abs/calib_0005_r4i.npy

           The space-separated format is mandated by RKNN Toolkit 2 for models
           with more than one input tensor.  Each position in the line maps 1:1
           to the model's input list in the order they appear in the ONNX graph
           (which matches the order in `input_shapes` dict in our YAML config).

        WHY AUXILIARY INPUTS ARE ZERO-INITIALIZED:
        -------------------------------------------
        For recurrent models, r1i~r4i are ConvGRU hidden states that carry
        temporal context from the *previous* frame.  During calibration there
        IS no previous frame -- RKNN quantizer feeds images one by one without
        running actual inference between them.  So zero-init is the only
        correct baseline: it matches the "first frame" condition where no
        temporal context exists yet.

        If INT8 quality is later found to drift over long video sequences (the
        quantizer never saw "warm" recurrent states), the mitigation path is
        hybrid precision: keep r1i~r4i in FP16 while the rest of the graph
        runs INT8.  That decision lives in Phase-4 Block 4.6, not here.
        """
        logger.info(f"Starting Vision calibration data generation...")
        logger.info(f"Dataset source: {dataset_path}")
        logger.info(f"Output directory: {output_dir}")

        # ----------------------------------------------------------------
        # STEP 1: Load the list of source image paths from the dataset file
        # ----------------------------------------------------------------
        image_paths = self._load_image_list(dataset_path)
        logger.info(f"Found {len(image_paths)} images in dataset")

        # ----------------------------------------------------------------
        # STEP 2: Read input_shapes from the YAML config
        # ----------------------------------------------------------------
        # The YAML looks like this for RVM:
        #
        #   input_shapes:
        #     src: [1, 3, 256, 256]
        #     r1i: [1, 16, 32, 32]
        #     r2i: [1, 20, 16, 16]
        #     r3i: [1, 40,  8,  8]
        #     r4i: [1, 64,  4,  4]
        #
        # And like this for MODNet:
        #
        #   input_shapes:
        #     - [1, 3, 512, 512]
        #
        # We need to handle both the old list format and new dict format.
        models = self.cfg.get('models', [])
        if not models:
            raise ValueError("No models found in configuration")

        input_shapes = models[0].get('input_shapes', [])
        if not input_shapes:
            raise ValueError("No input_shapes found in model configuration")

        # ----------------------------------------------------------------
        # STEP 3: Determine single-input vs multi-input
        # ----------------------------------------------------------------
        # This drives the entire branching logic below:
        #   - single-input: one .npy per sample, one path per line
        #   - multi-input:  N .npy per sample, space-separated paths per line
        multi_input = self._is_multi_input(input_shapes)

        # ----------------------------------------------------------------
        # STEP 4: Extract shapes for all inputs
        # ----------------------------------------------------------------
        # For dict format: preserve insertion order (Python 3.7+ guarantees
        # dict ordering).  The YAML must list `src` FIRST because RKNN maps
        # calibration file columns to model inputs by *position*.
        #
        # For list format: only the first (and only) entry is the image shape.
        if isinstance(input_shapes, dict):
            all_input_names = list(input_shapes.keys())
            all_input_shapes = list(input_shapes.values())
            primary_shape = all_input_shapes[0]
        else:
            all_input_names = None
            all_input_shapes = None
            primary_shape = input_shapes[0]

        # Sanity check: image input must be 4D (NCHW)
        if len(primary_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {primary_shape}")

        # Unpack NCHW → we need (H, W, C) for cv2.resize
        _, C, H, W = primary_shape
        target_shape = (H, W, C)

        logger.info(f"Model input shape: {primary_shape} (NCHW)")
        logger.info(f"Target preprocessing shape: {target_shape} (HWC)")

        if multi_input:
            logger.info(f"Multi-input model detected: {len(all_input_names)} inputs")
            for name, shape in zip(all_input_names, all_input_shapes):
                logger.info(f"  - {name}: {shape}")
            # Clarify which input gets real image data vs zero-fill:
            logger.info(
                f"  Primary input (image): '{all_input_names[0]}' "
                f"| Auxiliary inputs (zero-init): {all_input_names[1:]}"
            )

        # ----------------------------------------------------------------
        # STEP 5: Prepare output directory
        # ----------------------------------------------------------------
        # Use absolute path because RKNN Toolkit 2 resolves paths relative to
        # its own CWD, not ours.  Relative paths caused silent "file not found"
        # errors in earlier ArcFoundry versions.
        npy_dir = os.path.abspath(os.path.join(output_dir, "calibration_data"))
        ensure_dir(npy_dir)

        # ----------------------------------------------------------------
        # STEP 6: Main loop — preprocess images and save .npy files
        # ----------------------------------------------------------------
        # saved_lines collects one string per calibration sample.
        #   - single-input: each string is one absolute .npy path
        #   - multi-input:  each string is N space-separated absolute .npy paths
        saved_lines = []
        for idx, img_path in enumerate(
                tqdm(image_paths, desc="Processing images")):

            # Downsample the dataset by sampling_interval (default: every 5th).
            # RKNN recommends 100-300 calibration samples.  With 300 source
            # images and interval=5, we get 60 samples — within the sweet spot.
            if idx % self.sampling_interval != 0:
                continue

            try:
                # --- 6a. Preprocess the RGB image ---
                # Loads BGR via cv2, converts to RGB, resizes to (H,W),
                # casts to float32.  The pixel range stays 0-255 because
                # RKNN handles normalization internally via mean/std config.
                img_array = self._preprocess_image(img_path, target_shape)

                # --- 6b. Reshape from HWC to NCHW ---
                # RKNN expects calibration data in the same layout as model
                # input.  Our model input is NCHW (batch, channels, height,
                # width), so we transpose (H,W,C) → (C,H,W) then add batch.
                img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
                img_array = np.expand_dims(img_array, axis=0)   # CHW -> NCHW

                # ============================================================
                # BRANCH A: MULTI-INPUT MODEL (e.g. RVM)
                # ============================================================
                # For RVM, RKNN needs 5 .npy files per calibration sample:
                #   calib_0000_src.npy  ← real image data
                #   calib_0000_r1i.npy  ← zeros (no prior temporal context)
                #   calib_0000_r2i.npy  ← zeros
                #   calib_0000_r3i.npy  ← zeros
                #   calib_0000_r4i.npy  ← zeros
                #
                # The calibration_list.txt line for this sample:
                #   /abs/.../calib_0000_src.npy /abs/.../calib_0000_r1i.npy ...
                #
                # This space-separated format is how RKNN Toolkit 2 maps
                # each .npy to the corresponding model input by column index.
                # Column order MUST match `input_shapes` dict order in YAML.
                if multi_input:
                    step_files = []

                    # -- Save the primary (image) input --
                    # This is the only input that gets real data.
                    primary_name = all_input_names[0]
                    npy_filename = f"calib_{idx:04d}_{primary_name}.npy"
                    npy_path = os.path.join(npy_dir, npy_filename)
                    np.save(npy_path, img_array)
                    step_files.append(os.path.abspath(npy_path))

                    # -- Save auxiliary (recurrent state) inputs as zeros --
                    # WHY ZEROS:
                    #   The recurrent states r1i~r4i carry temporal memory from
                    #   previous frames.  During calibration, RKNN quantizer
                    #   does NOT run actual inference between samples — it just
                    #   feeds each sample independently to collect activation
                    #   statistics (min/max or KL histograms).
                    #
                    #   Therefore there is no "previous frame" to supply state
                    #   from.  Zero-init is correct: it matches the real first-
                    #   frame condition in production (RecurrentStateManager
                    #   initializes all states to zero on reset).
                    #
                    #   This means the quantizer only sees the "cold start"
                    #   distribution of recurrent channels, never the "warm"
                    #   steady-state distribution.  If this causes precision
                    #   drift in long videos, the fix is hybrid precision
                    #   (keep r1~r4 as FP16, quantize the rest to INT8).
                    #   That decision is made in Phase-4 Block 4.6, not here.
                    for aux_name, aux_shape in zip(
                            all_input_names[1:], all_input_shapes[1:]):
                        aux_data = np.zeros(aux_shape, dtype=np.float32)
                        aux_filename = f"calib_{idx:04d}_{aux_name}.npy"
                        aux_path = os.path.join(npy_dir, aux_filename)
                        np.save(aux_path, aux_data)
                        step_files.append(os.path.abspath(aux_path))

                    # Join all N paths with spaces — RKNN multi-input format.
                    # Example output line:
                    #   /abs/calib_0000_src.npy /abs/calib_0000_r1i.npy /abs/calib_0000_r2i.npy ...
                    saved_lines.append(" ".join(step_files))

                # ============================================================
                # BRANCH B: SINGLE-INPUT MODEL (e.g. MODNet, YOLO)
                # ============================================================
                # Original behavior: one .npy per sample, one path per line.
                # This path is completely unchanged from the pre-fix code.
                else:
                    npy_filename = f"calib_{idx:04d}.npy"
                    npy_path = os.path.join(npy_dir, npy_filename)
                    np.save(npy_path, img_array)
                    saved_lines.append(os.path.abspath(npy_path))

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                continue

        # ----------------------------------------------------------------
        # STEP 7: Summary logging
        # ----------------------------------------------------------------
        logger.info(f"Saved {len(saved_lines)} calibration samples")
        if multi_input:
            # For RVM with 5 inputs and 60 samples, this prints:
            #   "(300 total .npy files, 5 per sample)"
            logger.info(
                f"  ({len(saved_lines) * len(all_input_names)} total .npy files, "
                f"{len(all_input_names)} per sample)"
            )

        # ----------------------------------------------------------------
        # STEP 8: Write calibration_list.txt
        # ----------------------------------------------------------------
        # This file is what gets passed to rknn.build(dataset=...).
        #
        # Format rules enforced by RKNN Toolkit 2:
        #   - Single-input model: one absolute .npy path per line
        #   - Multi-input model:  N space-separated absolute paths per line,
        #     where N = number of model inputs, and column order matches the
        #     model's input order in the ONNX graph
        #
        # ABSOLUTE PATHS are critical: RKNN Toolkit resolves paths from its
        # own working directory, which may differ from ours.
        list_file_path = os.path.abspath(
            os.path.join(output_dir, "calibration_list.txt"))
        with open(list_file_path, 'w') as f:
            for line in saved_lines:
                f.write(f"{line}\n")

        logger.info(f"✅ Calibration list saved to: {list_file_path}")

        return list_file_path
