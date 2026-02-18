"""
Visualization utilities for Brain Tumor Detection project
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple
import cv2

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import PLOTS_DIR, CLASS_NAMES


class Visualizer:
    """
    Visualization class for training results and predictions
    """

    def __init__(self, save_dir: Path = PLOTS_DIR):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def plot_training_history(
        self,
        history,
        metrics: List[str] = ["accuracy", "loss"],
        save_name: str = "training_history.png"
    ):
        """
        Plot training history curves

        Args:
            history: Keras training history object or dict
            metrics: Metrics to plot
            save_name: Filename for saved plot
        """
        if hasattr(history, 'history'):
            history = history.history

        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))

        if num_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric in history:
                ax.plot(history[metric], label=f'Train {metric}', linewidth=2)
            if f'val_{metric}' in history:
                ax.plot(history[f'val_{metric}'], label=f'Val {metric}', linewidth=2)

            ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=12)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Training history saved to {self.save_dir / save_name}")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = CLASS_NAMES,
        normalize: bool = True,
        save_name: str = "confusion_matrix.png"
    ):
        """
        Plot confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            normalize: Whether to normalize the matrix
            save_name: Filename for saved plot
        """
        from sklearn.metrics import confusion_matrix

        # Convert one-hot to class indices if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            square=True,
            cbar_kws={'shrink': 0.8}
        )

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14)

        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Confusion matrix saved to {self.save_dir / save_name}")

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: List[str] = CLASS_NAMES,
        save_name: str = "roc_curves.png"
    ):
        """
        Plot ROC curves for each class

        Args:
            y_true: True labels (one-hot encoded)
            y_pred_proba: Predicted probabilities
            class_names: Names of classes
            save_name: Filename for saved plot
        """
        from sklearn.metrics import roc_curve, auc

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(class_names)))

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr, tpr,
                color=color,
                linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})'
            )

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] ROC curves saved to {self.save_dir / save_name}")

    def plot_sample_predictions(
        self,
        images: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: List[str] = CLASS_NAMES,
        num_samples: int = 16,
        save_name: str = "sample_predictions.png"
    ):
        """
        Plot grid of sample predictions

        Args:
            images: Input images
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            class_names: Names of classes
            num_samples: Number of samples to display
            save_name: Filename for saved plot
        """
        # Convert one-hot to indices if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        # Select random samples
        indices = np.random.choice(
            len(images),
            min(num_samples, len(images)),
            replace=False
        )

        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            ax = axes[i]
            img = images[idx]

            # Handle grayscale images
            if len(img.shape) == 2 or img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)

            true_label = class_names[y_true[idx]]
            pred_label = class_names[y_pred[idx]]
            confidence = y_pred_proba[idx].max() * 100

            # Color based on correctness
            color = 'green' if y_true[idx] == y_pred[idx] else 'red'

            ax.set_title(
                f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                color=color,
                fontsize=9
            )
            ax.axis('off')

        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')

        plt.suptitle('Sample Predictions', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Sample predictions saved to {self.save_dir / save_name}")

    def plot_class_distribution(
        self,
        labels: np.ndarray,
        class_names: List[str] = CLASS_NAMES,
        title: str = "Class Distribution",
        save_name: str = "class_distribution.png"
    ):
        """
        Plot class distribution bar chart

        Args:
            labels: Array of labels
            class_names: Names of classes
            title: Plot title
            save_name: Filename for saved plot
        """
        if len(labels.shape) > 1:
            labels = np.argmax(labels, axis=1)

        unique, counts = np.unique(labels, return_counts=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            [class_names[i] for i in unique],
            counts,
            color=sns.color_palette("husl", len(unique))
        )

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                str(count),
                ha='center',
                fontsize=10
            )

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Class distribution saved to {self.save_dir / save_name}")

    def plot_misclassified_samples(
        self,
        images: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: List[str] = CLASS_NAMES,
        num_samples: int = 20,
        save_name: str = "misclassified_samples.png"
    ):
        """
        Plot misclassified samples for error analysis

        Args:
            images: Input images
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            class_names: Names of classes
            num_samples: Number of samples to display
            save_name: Filename for saved plot
        """
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        # Find misclassified indices
        misclassified_idx = np.where(y_true != y_pred)[0]

        if len(misclassified_idx) == 0:
            print("No misclassified samples found!")
            return

        # Select samples
        num_to_show = min(num_samples, len(misclassified_idx))
        indices = np.random.choice(misclassified_idx, num_to_show, replace=False)

        grid_size = int(np.ceil(np.sqrt(num_to_show)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            ax = axes[i]
            img = images[idx]

            if len(img.shape) == 2 or img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)

            true_label = class_names[y_true[idx]]
            pred_label = class_names[y_pred[idx]]
            confidence = y_pred_proba[idx].max() * 100

            ax.set_title(
                f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                color='red',
                fontsize=9
            )
            ax.axis('off')

        for i in range(num_to_show, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Misclassified Samples ({len(misclassified_idx)} total)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Misclassified samples saved to {self.save_dir / save_name}")


if __name__ == "__main__":
    # Test visualization
    viz = Visualizer()

    # Create dummy data
    num_samples = 100
    num_classes = 4

    y_true = np.eye(num_classes)[np.random.randint(0, num_classes, num_samples)]
    y_pred_proba = np.random.rand(num_samples, num_classes)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    y_pred = np.eye(num_classes)[np.argmax(y_pred_proba, axis=1)]

    # Test confusion matrix
    viz.plot_confusion_matrix(y_true, y_pred)

    # Test ROC curves
    viz.plot_roc_curves(y_true, y_pred_proba)

    # Test class distribution
    viz.plot_class_distribution(y_true)

    print("\n[OK] All visualization tests passed!")
