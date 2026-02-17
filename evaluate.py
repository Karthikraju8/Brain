"""
Evaluation script for Brain Tumor Detection model

Usage:
    python evaluate.py --model_path models/saved/xfarmer_final.keras
"""

import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf

from config import (
    PROCESSED_DATA_DIR, SAVED_MODELS_DIR, CHECKPOINTS_DIR,
    CLASS_NAMES
)
from src.data.dataset import BrainTumorDataset
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import Visualizer
from src.utils.helpers import print_device_info


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Brain Tumor Detection model"
    )

    # Try to find a model automatically
    default_model = None
    for path in [
        SAVED_MODELS_DIR / "xfarmer_final.keras",
        CHECKPOINTS_DIR / "xfarmer_best.keras",
    ]:
        if path.exists():
            default_model = str(path)
            break

    parser.add_argument(
        "--model_path", type=str, default=default_model,
        help="Path to saved model file"
    )
    parser.add_argument(
        "--data_dir", type=str, default=str(PROCESSED_DATA_DIR),
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--save_predictions", action="store_true",
        help="Save predictions to file"
    )

    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()

    print("\n" + "=" * 60)
    print("BRAIN TUMOR DETECTION - MODEL EVALUATION")
    print("=" * 60)

    print_device_info()

    # ============== LOAD MODEL ==============
    print("\n[1/4] Loading Model...")
    print("-" * 40)

    if args.model_path is None:
        print("[WARN] ERROR: No model found!")
        print("Please specify a model path with --model_path")
        print("or train a model first using: python train.py")
        return

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[WARN] ERROR: Model not found at {model_path}")
        return

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[OK] Model loaded successfully")

    # Print model summary
    print(f"\nModel: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # ============== LOAD DATA ==============
    print("\n[2/4] Loading Test Data...")
    print("-" * 40)

    data_dir = Path(args.data_dir)

    if not (data_dir / "X_test.npy").exists():
        print(f"[WARN] ERROR: Processed data not found in {data_dir}")
        print("Please run training first: python train.py")
        return

    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    print(f"Test samples: {len(X_test)}")
    print(f"Test data shape: {X_test.shape}")

    # ============== EVALUATION ==============
    print("\n[3/4] Running Evaluation...")
    print("-" * 40)

    # Evaluate model
    results = model.evaluate(X_test, y_test, batch_size=args.batch_size, verbose=1)

    # Get metric names
    metric_names = model.metrics_names
    print("\nEvaluation Results:")
    for name, value in zip(metric_names, results):
        print(f"  {name}: {value:.4f}")

    # Get predictions
    print("\nGenerating predictions...")
    y_pred_proba = model.predict(X_test, batch_size=args.batch_size, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate detailed metrics
    print("\n[4/4] Calculating Metrics...")
    print("-" * 40)

    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(y_true, y_pred, y_pred_proba)
    metrics_calc.print_metrics(metrics)

    # Classification report
    print("\nClassification Report:")
    print("-" * 40)
    print(metrics_calc.get_classification_report(y_true, y_pred))

    # Save report
    metrics_calc.save_report(metrics, y_true, y_pred, "evaluation_report.txt")

    # ============== VISUALIZATIONS ==============
    print("\nGenerating Visualizations...")
    print("-" * 40)

    viz = Visualizer()

    # Confusion Matrix
    viz.plot_confusion_matrix(
        y_true, y_pred,
        save_name="eval_confusion_matrix.png"
    )

    # ROC Curves
    viz.plot_roc_curves(
        y_test, y_pred_proba,
        save_name="eval_roc_curves.png"
    )

    # Sample Predictions
    viz.plot_sample_predictions(
        X_test, y_test, y_pred, y_pred_proba,
        save_name="eval_sample_predictions.png"
    )

    # Misclassified Samples
    viz.plot_misclassified_samples(
        X_test, y_test, y_pred, y_pred_proba,
        save_name="eval_misclassified.png"
    )

    # ============== SAVE PREDICTIONS ==============
    if args.save_predictions:
        predictions_dir = Path("results") / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        np.save(predictions_dir / "y_pred.npy", y_pred)
        np.save(predictions_dir / "y_pred_proba.npy", y_pred_proba)
        np.save(predictions_dir / "y_true.npy", y_true)

        # Save predictions as CSV
        import pandas as pd
        df = pd.DataFrame({
            'true_label': [CLASS_NAMES[i] for i in y_true],
            'predicted_label': [CLASS_NAMES[i] for i in y_pred],
            'confidence': y_pred_proba.max(axis=1),
            **{f'prob_{name}': y_pred_proba[:, i] for i, name in enumerate(CLASS_NAMES)}
        })
        df.to_csv(predictions_dir / "predictions.csv", index=False)
        print(f"\n[OK] Predictions saved to {predictions_dir}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
