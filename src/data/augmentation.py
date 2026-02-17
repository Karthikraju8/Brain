"""
Data augmentation utilities for Brain Tumor MRI images
"""

import numpy as np
from typing import Tuple, Dict, Any
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import AUGMENTATION_CONFIG, IMAGE_SIZE, BATCH_SIZE


class DataAugmentor:
    """
    Data augmentation class for brain tumor MRI images
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize augmentor with configuration

        Args:
            config: Augmentation configuration dictionary
        """
        self.config = config or AUGMENTATION_CONFIG
        self.train_generator = None
        self.val_generator = None

    def create_generators(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = BATCH_SIZE
    ) -> Tuple[Any, Any]:
        """
        Create training and validation data generators

        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            batch_size: Batch size for generators

        Returns:
            Tuple of (train_generator, val_generator)
        """
        # Training generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=self.config.get("rotation_range", 20),
            width_shift_range=self.config.get("width_shift_range", 0.1),
            height_shift_range=self.config.get("height_shift_range", 0.1),
            shear_range=self.config.get("shear_range", 0.1),
            zoom_range=self.config.get("zoom_range", 0.1),
            horizontal_flip=self.config.get("horizontal_flip", True),
            vertical_flip=self.config.get("vertical_flip", False),
            fill_mode=self.config.get("fill_mode", "nearest"),
        )

        # Validation generator without augmentation
        val_datagen = ImageDataGenerator()

        self.train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )

        self.val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )

        return self.train_generator, self.val_generator

    def create_from_directory(
        self,
        train_dir: Path,
        val_dir: Path,
        batch_size: int = BATCH_SIZE,
        target_size: Tuple[int, int] = IMAGE_SIZE
    ) -> Tuple[Any, Any]:
        """
        Create generators that load images from directories

        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            batch_size: Batch size
            target_size: Target image size

        Returns:
            Tuple of (train_generator, val_generator)
        """
        # Training generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=self.config.get("rotation_range", 20),
            width_shift_range=self.config.get("width_shift_range", 0.1),
            height_shift_range=self.config.get("height_shift_range", 0.1),
            shear_range=self.config.get("shear_range", 0.1),
            zoom_range=self.config.get("zoom_range", 0.1),
            horizontal_flip=self.config.get("horizontal_flip", True),
            vertical_flip=self.config.get("vertical_flip", False),
            fill_mode=self.config.get("fill_mode", "nearest"),
        )

        # Validation generator
        val_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

        self.val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        return self.train_generator, self.val_generator

    def augment_image(self, image: np.ndarray, num_augmented: int = 5) -> np.ndarray:
        """
        Generate augmented versions of a single image

        Args:
            image: Input image (H, W, C)
            num_augmented: Number of augmented images to generate

        Returns:
            Array of augmented images (N, H, W, C)
        """
        datagen = ImageDataGenerator(
            rotation_range=self.config.get("rotation_range", 20),
            width_shift_range=self.config.get("width_shift_range", 0.1),
            height_shift_range=self.config.get("height_shift_range", 0.1),
            shear_range=self.config.get("shear_range", 0.1),
            zoom_range=self.config.get("zoom_range", 0.1),
            horizontal_flip=self.config.get("horizontal_flip", True),
            fill_mode=self.config.get("fill_mode", "nearest"),
        )

        # Reshape for generator
        image = image.reshape((1,) + image.shape)

        augmented_images = []
        for batch in datagen.flow(image, batch_size=1):
            augmented_images.append(batch[0])
            if len(augmented_images) >= num_augmented:
                break

        return np.array(augmented_images)


class AdvancedAugmentor:
    """
    Advanced augmentation using albumentations library
    """

    def __init__(self):
        try:
            import albumentations as A
            self.A = A
            self.available = True
        except ImportError:
            print("[WARN] albumentations not installed. Using basic augmentation.")
            self.available = False

    def get_train_transforms(self):
        """Get training augmentation pipeline"""
        if not self.available:
            return None

        return self.A.Compose([
            self.A.Rotate(limit=20, p=0.5),
            self.A.HorizontalFlip(p=0.5),
            self.A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            self.A.OneOf([
                self.A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                self.A.GaussianBlur(blur_limit=(3, 5), p=1),
            ], p=0.3),
            self.A.OneOf([
                self.A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1
                ),
                self.A.CLAHE(clip_limit=2.0, p=1),
            ], p=0.3),
            self.A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                fill_value=0,
                p=0.2
            ),
        ])

    def get_val_transforms(self):
        """Get validation transforms (no augmentation)"""
        if not self.available:
            return None

        return self.A.Compose([
            # Only normalization/resize if needed
        ])

    def apply_transforms(
        self,
        image: np.ndarray,
        transform
    ) -> np.ndarray:
        """Apply transformation to image"""
        if transform is None:
            return image

        augmented = transform(image=image)
        return augmented['image']


if __name__ == "__main__":
    # Test augmentation
    augmentor = DataAugmentor()

    # Create dummy data
    dummy_image = np.random.rand(224, 224, 3).astype(np.float32)

    # Test single image augmentation
    augmented = augmentor.augment_image(dummy_image, num_augmented=5)
    print(f"Original shape: {dummy_image.shape}")
    print(f"Augmented batch shape: {augmented.shape}")

    # Test advanced augmentor
    adv_augmentor = AdvancedAugmentor()
    if adv_augmentor.available:
        transforms = adv_augmentor.get_train_transforms()
        aug_img = adv_augmentor.apply_transforms(
            (dummy_image * 255).astype(np.uint8),
            transforms
        )
        print(f"Advanced augmented shape: {aug_img.shape}")
