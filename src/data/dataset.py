"""
Dataset loading and management for Brain Tumor MRI images
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE,
    NUM_CLASSES, CLASS_NAMES, TRAIN_SPLIT, VAL_SPLIT,
    TEST_SPLIT, RANDOM_SEED
)


class BrainTumorDataset:
    """
    Dataset class for loading and managing Brain Tumor MRI images

    Expected folder structure:
    data/raw/
        ├── glioma/
        │   ├── image1.jpg
        │   └── ...
        ├── meningioma/
        │   └── ...
        ├── notumor/
        │   └── ...
        └── pituitary/
            └── ...
    """

    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.image_size = IMAGE_SIZE
        self.class_names = CLASS_NAMES
        self.num_classes = NUM_CLASSES

        self.images = []
        self.labels = []

    def load_data(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all images from the data directory

        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        images = []
        labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name

            if not class_dir.exists():
                print(f"[WARN] Warning: Directory not found: {class_dir}")
                continue

            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))

            if verbose:
                print(f"Loading {class_name}: {len(image_files)} images")

            for img_path in tqdm(image_files, desc=class_name, disable=not verbose):
                img = self._load_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(class_idx)

        self.images = np.array(images, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int32)

        if verbose:
            print(f"\n[OK] Loaded {len(self.images)} images total")
            self._print_class_distribution()

        return self.images, self.labels

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """Load and resize a single image"""
        try:
            img = cv2.imread(str(path))
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            img = img / 255.0  # Normalize to [0, 1]
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def _print_class_distribution(self):
        """Print the distribution of classes in the dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\nClass Distribution:")
        print("-" * 30)
        for idx, count in zip(unique, counts):
            percentage = (count / len(self.labels)) * 100
            print(f"  {self.class_names[idx]}: {count} ({percentage:.1f}%)")

    def get_splits(
        self,
        one_hot: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets

        Args:
            one_hot: Whether to convert labels to one-hot encoding

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if len(self.images) == 0:
            self.load_data()

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, self.labels,
            test_size=TEST_SPLIT,
            random_state=RANDOM_SEED,
            stratify=self.labels
        )

        # Second split: separate train and validation
        val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=RANDOM_SEED,
            stratify=y_temp
        )

        if one_hot:
            y_train = to_categorical(y_train, self.num_classes)
            y_val = to_categorical(y_val, self.num_classes)
            y_test = to_categorical(y_test, self.num_classes)

        print(f"\n[OK] Data Split Complete:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_processed_data(self, output_dir: Path = PROCESSED_DATA_DIR):
        """Save processed data as numpy arrays"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X_train, X_val, X_test, y_train, y_val, y_test = self.get_splits()

        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "X_val.npy", X_val)
        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "y_val.npy", y_val)
        np.save(output_dir / "y_test.npy", y_test)

        print(f"[OK] Processed data saved to {output_dir}")

    @staticmethod
    def load_processed_data(
        data_dir: Path = PROCESSED_DATA_DIR
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """Load preprocessed data from numpy files"""
        data_dir = Path(data_dir)

        X_train = np.load(data_dir / "X_train.npy")
        X_val = np.load(data_dir / "X_val.npy")
        X_test = np.load(data_dir / "X_test.npy")
        y_train = np.load(data_dir / "y_train.npy")
        y_val = np.load(data_dir / "y_val.npy")
        y_test = np.load(data_dir / "y_test.npy")

        print(f"[OK] Loaded processed data from {data_dir}")
        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Test the dataset loader
    dataset = BrainTumorDataset()

    # Check if raw data exists
    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
        images, labels = dataset.load_data()
        dataset.save_processed_data()
    else:
        print(f"\n[WARN] No data found in {RAW_DATA_DIR}")
        print("Please download the dataset and place it in the data/raw/ folder")
        print("\nExpected structure:")
        print("  data/raw/glioma/")
        print("  data/raw/meningioma/")
        print("  data/raw/notumor/")
        print("  data/raw/pituitary/")
