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

    def run(self, onnx_path, output_dir, dataset_path):
        """
        Executes the calibration generation process for vision models.

        Args:
            onnx_path: Path to the source ONNX model
            output_dir: Workspace directory to save .npy files and the list
            dataset_path: Path to the text file containing image file paths

        Returns:
            str: Path to the generated calibration_list.txt
        """
        logger.info(f"Starting Vision calibration data generation...")
        logger.info(f"Dataset source: {dataset_path}")
        logger.info(f"Output directory: {output_dir}")

        # Load image list
        image_paths = self._load_image_list(dataset_path)
        logger.info(f"Found {len(image_paths)} images in dataset")

        # Get model input shape from config
        models = self.cfg.get('models', [])
        if not models:
            raise ValueError("No models found in configuration")

        input_shapes = models[0].get('input_shapes', [])
        if not input_shapes:
            raise ValueError("No input_shapes found in model configuration")

        # Support both list format (old) and dict format (new: {name: shape})
        if isinstance(input_shapes, dict):
            input_shape = list(input_shapes.values())[0]
        else:
            # Assume NCHW format: [batch, channels, height, width]
            input_shape = input_shapes[0]
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")

        _, C, H, W = input_shape
        target_shape = (H, W, C)

        logger.info(f"Model input shape: {input_shape} (NCHW)")
        logger.info(f"Target preprocessing shape: {target_shape} (HWC)")

        # Create output directory for .npy files
        # Use absolute path to avoid path duplication issues
        npy_dir = os.path.abspath(os.path.join(output_dir, "calibration_data"))
        ensure_dir(npy_dir)

        # Process images and save as .npy files
        saved_files = []
        for idx, img_path in enumerate(
                tqdm(image_paths, desc="Processing images")):
            # Apply sampling interval
            if idx % self.sampling_interval != 0:
                continue

            try:
                # Preprocess image
                img_array = self._preprocess_image(img_path, target_shape)

                # Add batch dimension and convert to NCHW format
                # img_array is currently (H, W, C), need to convert to (1, C, H, W)
                img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
                img_array = np.expand_dims(img_array, axis=0)  # CHW -> NCHW

                # Save as .npy file
                npy_filename = f"calib_{idx:04d}.npy"
                npy_path = os.path.join(npy_dir, npy_filename)
                np.save(npy_path, img_array)

                saved_files.append(npy_path)

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                continue

        logger.info(f"Saved {len(saved_files)} calibration samples")

        # Create calibration list file with absolute paths
        # IMPORTANT: Use absolute paths to avoid path duplication issues with RKNN toolkit
        list_file_path = os.path.abspath(
            os.path.join(output_dir, "calibration_list.txt"))
        with open(list_file_path, 'w') as f:
            for npy_path in saved_files:
                # Ensure all paths are absolute
                abs_npy_path = os.path.abspath(npy_path)
                f.write(f"{abs_npy_path}\n")

        logger.info(f"✅ Calibration list saved to: {list_file_path}")

        return list_file_path
