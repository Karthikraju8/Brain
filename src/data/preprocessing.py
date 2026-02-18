"""
Data preprocessing utilities for Brain Tumor MRI images
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import IMAGE_SIZE


class DataPreprocessor:
    """
    Preprocessing pipeline for MRI brain images
    """

    def __init__(self, target_size: Tuple[int, int] = IMAGE_SIZE):
        self.target_size = target_size

    def preprocess_image(
        self,
        image: np.ndarray,
        denoise: bool = True,
        normalize: bool = True,
        enhance_contrast: bool = True
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline to a single image

        Args:
            image: Input image (BGR or RGB)
            denoise: Apply denoising
            normalize: Normalize pixel values
            enhance_contrast: Apply CLAHE contrast enhancement

        Returns:
            Preprocessed image
        """
        # Resize
        img = cv2.resize(image, self.target_size)

        # Convert to grayscale for processing, then back to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Denoise
        if denoise:
            gray = self.apply_denoising(gray)

        # Enhance contrast
        if enhance_contrast:
            gray = self.apply_clahe(gray)

        # Convert back to 3 channels
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Normalize
        if normalize:
            img = self.normalize(img)

        return img

    def apply_denoising(
        self,
        image: np.ndarray,
        method: str = "gaussian"
    ) -> np.ndarray:
        """
        Apply denoising to image

        Args:
            image: Input image
            method: Denoising method ('gaussian', 'median', 'bilateral')
        """
        if method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(image, 5)
        elif method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            return image

    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

        Args:
            image: Input grayscale image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        return clahe.apply(image)

    def normalize(
        self,
        image: np.ndarray,
        method: str = "minmax"
    ) -> np.ndarray:
        """
        Normalize image pixel values

        Args:
            image: Input image
            method: Normalization method ('minmax', 'zscore')
        """
        image = image.astype(np.float32)

        if method == "minmax":
            # Scale to [0, 1]
            return image / 255.0
        elif method == "zscore":
            # Zero mean, unit variance
            mean = np.mean(image)
            std = np.std(image)
            return (image - mean) / (std + 1e-7)
        else:
            return image

    def crop_brain_region(
        self,
        image: np.ndarray,
        threshold: int = 10
    ) -> np.ndarray:
        """
        Crop to the brain region by removing black borders

        Args:
            image: Input image
            threshold: Pixel intensity threshold for detecting brain region
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Threshold to get binary mask
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Get the largest contour (brain region)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)

            cropped = image[y:y+h, x:x+w]
            return cv2.resize(cropped, self.target_size)

        return cv2.resize(image, self.target_size)

    def preprocess_batch(
        self,
        images: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Preprocess a batch of images

        Args:
            images: Batch of images (N, H, W, C)
            **kwargs: Arguments passed to preprocess_image
        """
        processed = []
        for img in images:
            processed.append(self.preprocess_image(img, **kwargs))
        return np.array(processed)


class MRISpecificPreprocessor(DataPreprocessor):
    """
    Specialized preprocessor for MRI brain images
    """

    def skull_stripping(
        self,
        image: np.ndarray,
        threshold: int = 30
    ) -> np.ndarray:
        """
        Simple skull stripping using thresholding and morphological operations
        Note: For production, use dedicated tools like FSL BET
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Apply mask
        result = cv2.bitwise_and(image, image, mask=binary)

        return result

    def intensity_standardization(
        self,
        image: np.ndarray,
        target_mean: float = 0.5,
        target_std: float = 0.2
    ) -> np.ndarray:
        """
        Standardize intensity values across images
        """
        image = image.astype(np.float32)

        # Get non-zero pixels (brain region)
        mask = image > 0.01

        if np.sum(mask) > 0:
            current_mean = np.mean(image[mask])
            current_std = np.std(image[mask])

            if current_std > 0:
                # Standardize
                image[mask] = (image[mask] - current_mean) / current_std
                # Scale to target
                image[mask] = image[mask] * target_std + target_mean
                # Clip to valid range
                image = np.clip(image, 0, 1)

        return image


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = MRISpecificPreprocessor()

    # Create a dummy image for testing
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Test preprocessing pipeline
    processed = preprocessor.preprocess_image(dummy_image)
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Output range: [{processed.min():.3f}, {processed.max():.3f}]")
