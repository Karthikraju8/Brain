"""
Training callbacks for the X-Farmer model
"""

import os
from pathlib import Path
from typing import List
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger,
    Callback
)

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    CHECKPOINTS_DIR, LOGS_DIR, TENSORBOARD_LOG_DIR,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR, MIN_LR
)


class TrainingMonitor(Callback):
    """
    Custom callback to monitor training progress
    """

    def __init__(self, print_freq: int = 1):
        super().__init__()
        self.print_freq = print_freq
        self.best_val_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_accuracy = logs.get("val_accuracy", 0)

        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            print(f"\n[*] New best validation accuracy: {val_accuracy:.4f}")

        if (epoch + 1) % self.print_freq == 0:
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {logs.get('loss', 0):.4f}")
            print(f"  Train Accuracy: {logs.get('accuracy', 0):.4f}")
            print(f"  Val Loss: {logs.get('val_loss', 0):.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            print(f"  Learning Rate: {logs.get('lr', 0):.6f}")


def get_callbacks(
    model_name: str = "xfarmer",
    monitor: str = "val_accuracy",
    mode: str = "max",
    use_tensorboard: bool = True,
    use_csv_logger: bool = True,
    custom_callbacks: List[Callback] = None
) -> List[Callback]:
    """
    Get list of callbacks for model training

    Args:
        model_name: Name for saving checkpoints
        monitor: Metric to monitor for checkpointing
        mode: 'min' or 'max' for the monitored metric
        use_tensorboard: Whether to use TensorBoard logging
        use_csv_logger: Whether to log training history to CSV
        custom_callbacks: Additional custom callbacks

    Returns:
        List of Keras callbacks
    """
    callbacks = []

    # Model checkpoint - save best model
    checkpoint_path = CHECKPOINTS_DIR / f"{model_name}_best.keras"
    checkpoint_callback = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # Save latest model (for resume training)
    latest_checkpoint_path = CHECKPOINTS_DIR / f"{model_name}_latest.keras"
    latest_checkpoint = ModelCheckpoint(
        filepath=str(latest_checkpoint_path),
        monitor=monitor,
        mode=mode,
        save_best_only=False,
        save_weights_only=False,
        verbose=0
    )
    callbacks.append(latest_checkpoint)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # TensorBoard logging
    if use_tensorboard:
        tensorboard_callback = TensorBoard(
            log_dir=str(TENSORBOARD_LOG_DIR),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch"
        )
        callbacks.append(tensorboard_callback)

    # CSV Logger
    if use_csv_logger:
        csv_path = LOGS_DIR / f"{model_name}_training_log.csv"
        csv_logger = CSVLogger(
            str(csv_path),
            separator=",",
            append=True
        )
        callbacks.append(csv_logger)

    # Custom training monitor
    callbacks.append(TrainingMonitor(print_freq=5))

    # Add any custom callbacks
    if custom_callbacks:
        callbacks.extend(custom_callbacks)

    return callbacks


class GradCAMCallback(Callback):
    """
    Callback to generate Grad-CAM visualizations during training
    """

    def __init__(
        self,
        validation_data,
        layer_name: str,
        output_dir: Path,
        num_samples: int = 5,
        freq: int = 10
    ):
        super().__init__()
        self.validation_data = validation_data
        self.layer_name = layer_name
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.freq = freq

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            # Generate Grad-CAM visualizations
            # Implementation would go here
            print(f"\n[GradCAM] Generating visualizations for epoch {epoch + 1}")


if __name__ == "__main__":
    # Test callbacks
    callbacks = get_callbacks(model_name="xfarmer_test")
    print(f"Created {len(callbacks)} callbacks:")
    for cb in callbacks:
        print(f"  - {cb.__class__.__name__}")
