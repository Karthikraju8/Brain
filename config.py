"""
Configuration file for Brain Tumor Detection Project
Contains all hyperparameters and settings
"""

import os
from pathlib import Path

# ============== PATH CONFIGURATION ==============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"

MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
SAVED_MODELS_DIR = MODELS_DIR / "saved"

LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"

# ============== DATA CONFIGURATION ==============
IMAGE_SIZE = (224, 224)
CHANNELS = 3  # RGB
NUM_CLASSES = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# Data split ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============== MODEL CONFIGURATION ==============
MODEL_NAME = "xfarmer"

# X-Farmer Architecture Config
XFARMER_CONFIG = {
    "conv_filters": [32, 64, 128, 256],
    "kernel_size": (3, 3),
    "pool_size": (2, 2),
    "dense_units": [512, 256],
    "dropout_rate": 0.5,
    "use_batch_norm": True,
}

# ============== TRAINING CONFIGURATION ==============
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Optimizer
OPTIMIZER = "adam"  # Options: "adam", "sgd", "rmsprop"

# Loss function
LOSS_FUNCTION = "categorical_crossentropy"

# ============== AUGMENTATION CONFIGURATION ==============
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "shear_range": 0.1,
    "zoom_range": 0.1,
    "horizontal_flip": True,
    "vertical_flip": False,
    "fill_mode": "nearest",
}

# ============== INFERENCE CONFIGURATION ==============
CONFIDENCE_THRESHOLD = 0.5

# ============== LOGGING CONFIGURATION ==============
LOG_LEVEL = "INFO"
TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"

# ============== DEVICE CONFIGURATION ==============
DEVICE = "AUTO"  # Options: "AUTO", "CPU", "GPU" — AUTO detects GPU automatically
GPU_MEMORY_GROWTH = True
GPU_MEMORY_LIMIT = None  # Set in MB to limit GPU memory usage, None for no limit
MIXED_PRECISION = False  # Enable mixed precision (FP16) for faster GPU training

# ============== RANDOM SEED ==============
RANDOM_SEED = 42


def create_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, AUGMENTED_DATA_DIR,
        MODELS_DIR, CHECKPOINTS_DIR, SAVED_MODELS_DIR,
        LOGS_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR,
        TENSORBOARD_LOG_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
    print("[OK] All directories created successfully!")
