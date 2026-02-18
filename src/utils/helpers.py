"""
Helper utilities for the Brain Tumor Detection project
"""

import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RANDOM_SEED, DEVICE, GPU_MEMORY_GROWTH, GPU_MEMORY_LIMIT, MIXED_PRECISION


def set_seed(seed: int = RANDOM_SEED):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # For GPU determinism (may impact performance)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    print(f"[OK] Random seed set to {seed}")


def get_device_info() -> dict:
    """
    Get information about available compute devices

    Returns:
        Dictionary with device information
    """
    info = {
        "tensorflow_version": tf.__version__,
        "gpus_available": [],
        "using_gpu": False
    }

    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        info["gpus_available"] = [gpu.name for gpu in gpus]
        info["using_gpu"] = True
        info["num_gpus"] = len(gpus)

        # Get GPU details
        try:
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                info["gpu_details"] = details
        except Exception:
            pass
    else:
        info["num_gpus"] = 0

    return info


def print_device_info():
    """Print device information in formatted way"""
    info = get_device_info()

    print("\n" + "=" * 40)
    print("SYSTEM INFORMATION")
    print("=" * 40)
    print(f"TensorFlow Version: {info['tensorflow_version']}")
    print(f"GPUs Available: {info['num_gpus']}")

    if info['using_gpu']:
        print("GPU Devices:")
        for gpu in info['gpus_available']:
            print(f"  - {gpu}")
    else:
        print("Running on CPU")

    print("=" * 40 + "\n")


def configure_gpu(memory_growth: bool = GPU_MEMORY_GROWTH, memory_limit: Optional[int] = GPU_MEMORY_LIMIT):
    """
    Configure device settings based on config.DEVICE

    Supports AUTO mode which detects GPU availability automatically.

    Args:
        memory_growth: Enable memory growth (allocate as needed)
        memory_limit: Limit GPU memory usage in MB
    """
    device_mode = DEVICE.upper()

    if device_mode == "CPU":
        # Force CPU usage by hiding all GPUs
        tf.config.set_visible_devices([], 'GPU')
        print("[INFO] Configured to use CPU (DEVICE='CPU' in config)")
        return

    # AUTO or GPU mode — detect available GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        if device_mode == "GPU":
            print("[WARN] DEVICE is set to 'GPU' but no GPU found! Falling back to CPU.")
            print("[WARN] Make sure CUDA and cuDNN are installed correctly.")
        else:
            print("[INFO] No GPU detected — using CPU for training.")
        return

    try:
        for gpu in gpus:
            if memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)

            if memory_limit:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit
                    )]
                )

        # Enable mixed precision for faster GPU training if configured
        if MIXED_PRECISION:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("[OK] Mixed precision (FP16) enabled")

        print(f"[OK] GPU detected and configured: {len(gpus)} GPU(s) available (memory_growth={memory_growth})")
    except RuntimeError as e:
        print(f"[WARN] GPU configuration error: {e}. Falling back to CPU.")


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_dir(base_dir: Path, experiment_name: str = None) -> Path:
    """
    Create a directory for experiment outputs

    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name

    Returns:
        Path to experiment directory
    """
    timestamp = get_timestamp()

    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = f"experiment_{timestamp}"

    experiment_dir = Path(base_dir) / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OK] Created experiment directory: {experiment_dir}")
    return experiment_dir


def count_parameters(model) -> dict:
    """
    Count trainable and non-trainable parameters in a model

    Args:
        model: Keras model

    Returns:
        Dictionary with parameter counts
    """
    trainable = np.sum([
        np.prod(v.get_shape().as_list())
        for v in model.trainable_weights
    ])

    non_trainable = np.sum([
        np.prod(v.get_shape().as_list())
        for v in model.non_trainable_weights
    ])

    total = trainable + non_trainable

    return {
        "trainable": int(trainable),
        "non_trainable": int(non_trainable),
        "total": int(total)
    }


def format_params(num_params: int) -> str:
    """Format number of parameters for display"""
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


class Timer:
    """Simple timer context manager"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time
        print(f"[TIME] {self.name} completed in {duration}")

    @property
    def elapsed(self):
        if self.start_time is None:
            return None
        end = self.end_time or datetime.now()
        return end - self.start_time


if __name__ == "__main__":
    # Test utilities
    print("\nTesting helper utilities...\n")

    # Set seed
    set_seed()

    # Print device info
    print_device_info()

    # Test timer
    with Timer("Test operation"):
        import time
        time.sleep(1)

    print("\n[OK] All helper tests passed!")
