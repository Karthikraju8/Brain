"""
Metrics calculation utilities for model evaluation
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef
)

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CLASS_NAMES, REPORTS_DIR


class MetricsCalculator:
    """
    Calculate and report evaluation metrics for classification
    """

    def __init__(self, class_names: list = CLASS_NAMES):
        self.class_names = class_names
        self.num_classes = len(class_names)

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate all evaluation metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for AUC)

        Returns:
            Dictionary of metrics
        """
        # Convert one-hot to indices if needed
        if len(y_true.shape) > 1:
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            y_true_idx = y_true

        if len(y_pred.shape) > 1:
            y_pred_idx = np.argmax(y_pred, axis=1)
        else:
            y_pred_idx = y_pred

        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true_idx, y_pred_idx)

        # Per-class metrics
        metrics['precision_macro'] = precision_score(
            y_true_idx, y_pred_idx, average='macro', zero_division=0
        )
        metrics['precision_weighted'] = precision_score(
            y_true_idx, y_pred_idx, average='weighted', zero_division=0
        )

        metrics['recall_macro'] = recall_score(
            y_true_idx, y_pred_idx, average='macro', zero_division=0
        )
        metrics['recall_weighted'] = recall_score(
            y_true_idx, y_pred_idx, average='weighted', zero_division=0
        )

        metrics['f1_macro'] = f1_score(
            y_true_idx, y_pred_idx, average='macro', zero_division=0
        )
        metrics['f1_weighted'] = f1_score(
            y_true_idx, y_pred_idx, average='weighted', zero_division=0
        )

        # Additional metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true_idx, y_pred_idx)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true_idx, y_pred_idx)

        # AUC (if probabilities provided)
        if y_pred_proba is not None:
            try:
                # One-hot encode y_true if needed
                if len(y_true.shape) == 1:
                    y_true_onehot = np.eye(self.num_classes)[y_true]
                else:
                    y_true_onehot = y_true

                metrics['auc_macro'] = roc_auc_score(
                    y_true_onehot, y_pred_proba, average='macro', multi_class='ovr'
                )
                metrics['auc_weighted'] = roc_auc_score(
                    y_true_onehot, y_pred_proba, average='weighted', multi_class='ovr'
                )
            except Exception as e:
                print(f"Warning: Could not calculate AUC: {e}")
                metrics['auc_macro'] = None
                metrics['auc_weighted'] = None

        # Per-class metrics
        precision_per_class = precision_score(
            y_true_idx, y_pred_idx, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true_idx, y_pred_idx, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true_idx, y_pred_idx, average=None, zero_division=0
        )

        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics['per_class'][class_name] = {
                    'precision': precision_per_class[i],
                    'recall': recall_per_class[i],
                    'f1': f1_per_class[i]
                }

        return metrics

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dict: bool = False
    ):
        """
        Get sklearn classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dict: Return as dictionary

        Returns:
            Classification report (string or dict)
        """
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=output_dict,
            zero_division=0
        )

    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Get confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalize by true labels

        Returns:
            Confusion matrix array
        """
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        return cm

    def print_metrics(self, metrics: Dict):
        """Print metrics in a formatted way"""
        print("\n" + "=" * 50)
        print("EVALUATION METRICS")
        print("=" * 50)

        print(f"\n{'Metric':<25} {'Value':>10}")
        print("-" * 35)

        main_metrics = [
            ('Accuracy', 'accuracy'),
            ('Precision (Macro)', 'precision_macro'),
            ('Precision (Weighted)', 'precision_weighted'),
            ('Recall (Macro)', 'recall_macro'),
            ('Recall (Weighted)', 'recall_weighted'),
            ('F1-Score (Macro)', 'f1_macro'),
            ('F1-Score (Weighted)', 'f1_weighted'),
            ('Cohen Kappa', 'cohen_kappa'),
            ('Matthews Corr.', 'matthews_corrcoef'),
            ('AUC (Macro)', 'auc_macro'),
            ('AUC (Weighted)', 'auc_weighted'),
        ]

        for name, key in main_metrics:
            value = metrics.get(key)
            if value is not None:
                print(f"{name:<25} {value:>10.4f}")

        print("\n" + "-" * 50)
        print("Per-Class Metrics:")
        print("-" * 50)

        if 'per_class' in metrics:
            print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
            print("-" * 50)
            for class_name, class_metrics in metrics['per_class'].items():
                print(
                    f"{class_name:<15} "
                    f"{class_metrics['precision']:>10.4f} "
                    f"{class_metrics['recall']:>10.4f} "
                    f"{class_metrics['f1']:>10.4f}"
                )

        print("=" * 50)

    def save_report(
        self,
        metrics: Dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str = "evaluation_report.txt",
        output_dir: Path = REPORTS_DIR
    ):
        """
        Save evaluation report to file

        Args:
            metrics: Calculated metrics
            y_true: True labels
            y_pred: Predicted labels
            filename: Output filename
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("BRAIN TUMOR CLASSIFICATION - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("OVERALL METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy:            {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (Macro):   {metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro):      {metrics['recall_macro']:.4f}\n")
            f.write(f"F1-Score (Macro):    {metrics['f1_macro']:.4f}\n")
            f.write(f"Cohen Kappa:         {metrics['cohen_kappa']:.4f}\n")

            if metrics.get('auc_macro'):
                f.write(f"AUC (Macro):         {metrics['auc_macro']:.4f}\n")

            f.write("\n\nCLASSIFICATION REPORT\n")
            f.write("-" * 40 + "\n")
            f.write(self.get_classification_report(y_true, y_pred))

            f.write("\n\nCONFUSION MATRIX\n")
            f.write("-" * 40 + "\n")
            cm = self.get_confusion_matrix(y_true, y_pred)
            f.write(str(cm))

        print(f"[OK] Report saved to {filepath}")


if __name__ == "__main__":
    # Test metrics calculation
    calculator = MetricsCalculator()

    # Create dummy data
    num_samples = 100
    num_classes = 4

    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred = np.random.randint(0, num_classes, num_samples)
    y_pred_proba = np.random.rand(num_samples, num_classes)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

    # Calculate metrics
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_pred_proba)

    # Print metrics
    calculator.print_metrics(metrics)

    # Save report
    calculator.save_report(metrics, y_true, y_pred)
