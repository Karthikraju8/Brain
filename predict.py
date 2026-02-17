"""
Inference script for Brain Tumor Detection

Usage:
    python predict.py --image path/to/mri_image.jpg
    python predict.py --image_dir path/to/images/
"""

import argparse
import numpy as np
from pathlib import Path
import cv2
import tensorflow as tf

from config import (
    SAVED_MODELS_DIR, CHECKPOINTS_DIR,
    IMAGE_SIZE, CLASS_NAMES, CONFIDENCE_THRESHOLD
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Predict brain tumor type from MRI image"
    )

    # Try to find model automatically
    default_model = None
    for path in [
        SAVED_MODELS_DIR / "xfarmer_final.keras",
        CHECKPOINTS_DIR / "xfarmer_best.keras",
    ]:
        if path.exists():
            default_model = str(path)
            break

    parser.add_argument(
        "--image", type=str,
        help="Path to single MRI image"
    )
    parser.add_argument(
        "--image_dir", type=str,
        help="Path to directory of MRI images"
    )
    parser.add_argument(
        "--model_path", type=str, default=default_model,
        help="Path to saved model"
    )
    parser.add_argument(
        "--threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display images with predictions"
    )

    return parser.parse_args()


def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    """
    Load and preprocess a single image for inference

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image array
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, IMAGE_SIZE)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img


def predict_single_image(
    model,
    image_path: Path,
    threshold: float = CONFIDENCE_THRESHOLD
) -> dict:
    """
    Make prediction for a single image

    Args:
        model: Loaded Keras model
        image_path: Path to image
        threshold: Confidence threshold

    Returns:
        Dictionary with prediction results
    """
    # Load and preprocess
    img = load_and_preprocess_image(image_path)

    # Add batch dimension
    img_batch = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img_batch, verbose=0)[0]

    # Get predicted class
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]

    result = {
        "image_path": str(image_path),
        "predicted_class": CLASS_NAMES[predicted_class_idx],
        "predicted_class_idx": int(predicted_class_idx),
        "confidence": float(confidence),
        "probabilities": {
            name: float(prob)
            for name, prob in zip(CLASS_NAMES, predictions)
        },
        "high_confidence": confidence >= threshold
    }

    return result


def display_prediction(image_path: Path, result: dict):
    """Display image with prediction overlay"""
    import matplotlib.pyplot as plt

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Show image
    ax1.imshow(img)
    ax1.set_title(f"Predicted: {result['predicted_class']}\n"
                  f"Confidence: {result['confidence']*100:.1f}%",
                  fontsize=12)
    ax1.axis('off')

    # Show probability bar chart
    probs = list(result['probabilities'].values())
    colors = ['green' if i == result['predicted_class_idx'] else 'steelblue'
              for i in range(len(CLASS_NAMES))]

    bars = ax2.barh(CLASS_NAMES, probs, color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probabilities')

    # Add percentage labels
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center')

    plt.tight_layout()
    plt.show()


def main():
    """Main inference function"""
    args = parse_args()

    print("\n" + "=" * 60)
    print("BRAIN TUMOR DETECTION - INFERENCE")
    print("=" * 60)

    # Validate inputs
    if not args.image and not args.image_dir:
        print("[WARN] ERROR: Please provide --image or --image_dir")
        return

    if args.model_path is None:
        print("[WARN] ERROR: No model found!")
        print("Please specify a model path with --model_path")
        return

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[WARN] ERROR: Model not found at {model_path}")
        return

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[OK] Model loaded successfully\n")

    # Collect images to process
    image_paths = []

    if args.image:
        image_path = Path(args.image)
        if image_path.exists():
            image_paths.append(image_path)
        else:
            print(f"[WARN] Image not found: {image_path}")
            return

    if args.image_dir:
        image_dir = Path(args.image_dir)
        if image_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend(image_dir.glob(ext))
        else:
            print(f"[WARN] Directory not found: {image_dir}")
            return

    if not image_paths:
        print("[WARN] No images found to process")
        return

    print(f"Processing {len(image_paths)} image(s)...\n")
    print("-" * 60)

    # Process each image
    results = []

    for image_path in image_paths:
        try:
            result = predict_single_image(model, image_path, args.threshold)
            results.append(result)

            # Print result
            print(f"\n[IMG] {image_path.name}")
            print(f"   Prediction: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']*100:.1f}%")

            if not result['high_confidence']:
                print(f"   [WARN] Low confidence (below {args.threshold*100:.0f}%)")

            # Show image if requested
            if args.show:
                display_prediction(image_path, result)

        except Exception as e:
            print(f"\n[WARN] Error processing {image_path}: {e}")

    # Summary
    print("\n" + "-" * 60)
    print("\nSUMMARY")
    print("-" * 30)

    if results:
        # Count predictions by class
        class_counts = {}
        for result in results:
            cls = result['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        print("Predictions by class:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")

        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\nAverage confidence: {avg_confidence*100:.1f}%")

        # Low confidence count
        low_conf_count = sum(1 for r in results if not r['high_confidence'])
        if low_conf_count > 0:
            print(f"Low confidence predictions: {low_conf_count}")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
