"""
Training script for Brain Tumor Detection using X-Farmer Model

Usage:
    python train.py
    python train.py --epochs 50 --batch_size 16
"""

import argparse
import numpy as np
from pathlib import Path

# Project imports
from config import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED,
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR,
    create_directories
)
from src.data.dataset import BrainTumorDataset
from src.data.augmentation import DataAugmentor
from src.models.xfarmer import XFarmerModel, build_xfarmer_model
from src.models.callbacks import get_callbacks
from src.utils.helpers import set_seed, print_device_info, configure_gpu, Timer
from src.utils.visualization import Visualizer
from src.utils.metrics import MetricsCalculator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train X-Farmer model for Brain Tumor Detection"
    )

    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--model", type=str, default="xfarmer",
        choices=["xfarmer", "xfarmer_lite"],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--augment", action="store_true", default=True,
        help="Use data augmentation"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "--load_processed", action="store_true",
        help="Load preprocessed data from numpy files"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Setup
    print("\n" + "=" * 60)
    print("BRAIN TUMOR DETECTION - X-FARMER MODEL TRAINING")
    print("=" * 60)

    # Create directories
    create_directories()

    # Set random seed
    set_seed(args.seed)

    # Configure GPU
    configure_gpu(memory_growth=True)

    # Print device info
    print_device_info()

    # ============== DATA LOADING ==============
    print("\n[1/5] Loading Data...")
    print("-" * 40)

    if args.load_processed and (PROCESSED_DATA_DIR / "X_train.npy").exists():
        # Load preprocessed data
        print("Loading preprocessed data from numpy files...")
        X_train, X_val, X_test, y_train, y_val, y_test = \
            BrainTumorDataset.load_processed_data()
    else:
        # Load and preprocess from raw images
        dataset = BrainTumorDataset(RAW_DATA_DIR)

        # Check if data exists
        if not RAW_DATA_DIR.exists() or not any(RAW_DATA_DIR.iterdir()):
            print(f"\n[WARN] ERROR: No data found in {RAW_DATA_DIR}")
            print("\nPlease download the dataset and organize it as follows:")
            print(f"  {RAW_DATA_DIR}/glioma/")
            print(f"  {RAW_DATA_DIR}/meningioma/")
            print(f"  {RAW_DATA_DIR}/notumor/")
            print(f"  {RAW_DATA_DIR}/pituitary/")
            print("\nDataset sources:")
            print("  - Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
            return

        images, labels = dataset.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_splits()

        # Save processed data for faster loading next time
        dataset.save_processed_data()

    print(f"\nDataset shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    # ============== DATA AUGMENTATION ==============
    print("\n[2/5] Setting up Data Augmentation...")
    print("-" * 40)

    if args.augment:
        augmentor = DataAugmentor()
        train_generator, val_generator = augmentor.create_generators(
            X_train, y_train, X_val, y_val,
            batch_size=args.batch_size
        )
        print("[OK] Data augmentation enabled")
    else:
        train_generator = None
        val_generator = None
        print("[OK] Data augmentation disabled")

    # ============== MODEL BUILDING ==============
    print("\n[3/5] Building Model...")
    print("-" * 40)

    if args.model == "xfarmer_lite":
        from src.models.xfarmer import build_xfarmer_lite
        model = build_xfarmer_lite()
        model_name = "xfarmer_lite"
    else:
        xfarmer = XFarmerModel()
        model = xfarmer.build()
        xfarmer.compile(learning_rate=args.lr)
        model_name = "xfarmer"

    model.summary()

    # ============== TRAINING ==============
    print("\n[4/5] Training Model...")
    print("-" * 40)

    callbacks = get_callbacks(model_name=model_name)

    with Timer("Training"):
        if args.augment and train_generator is not None:
            # Train with data augmentation
            history = model.fit(
                train_generator,
                steps_per_epoch=len(X_train) // args.batch_size,
                epochs=args.epochs,
                validation_data=val_generator,
                validation_steps=len(X_val) // args.batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without augmentation
            history = model.fit(
                X_train, y_train,
                batch_size=args.batch_size,
                epochs=args.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

    # ============== EVALUATION ==============
    print("\n[5/5] Evaluating Model...")
    print("-" * 40)

    # Evaluate on test set
    test_results = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")

    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate metrics
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(y_true, y_pred, y_pred_proba)
    metrics_calc.print_metrics(metrics)
    metrics_calc.save_report(metrics, y_true, y_pred)

    # Visualizations
    viz = Visualizer()
    viz.plot_training_history(history)
    viz.plot_confusion_matrix(y_true, y_pred)
    viz.plot_roc_curves(y_test, y_pred_proba)
    viz.plot_sample_predictions(X_test, y_test, y_pred, y_pred_proba)
    viz.plot_misclassified_samples(X_test, y_test, y_pred, y_pred_proba)

    # Save final model
    final_model_path = SAVED_MODELS_DIR / f"{model_name}_final.keras"
    model.save(final_model_path)
    print(f"\n[OK] Final model saved to {final_model_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
