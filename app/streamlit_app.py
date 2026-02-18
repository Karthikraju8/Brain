"""
Streamlit Web Application for Brain Tumor Detection

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from PIL import Image
import io

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    SAVED_MODELS_DIR, CHECKPOINTS_DIR,
    IMAGE_SIZE, CLASS_NAMES
)

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .tumor-detected {
        background-color: #FFEBEE;
        border: 2px solid #E53935;
    }
    .no-tumor {
        background-color: #E8F5E9;
        border: 2px solid #43A047;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cached)"""
    model_paths = [
        SAVED_MODELS_DIR / "xfarmer_final.keras",
        CHECKPOINTS_DIR / "xfarmer_best.keras",
    ]

    for path in model_paths:
        if path.exists():
            return tf.keras.models.load_model(path)

    return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess uploaded image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array
    img_array = np.array(image)

    # Resize
    img_resized = cv2.resize(img_array, IMAGE_SIZE)

    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0

    return img_normalized


def make_prediction(model, image: np.ndarray) -> dict:
    """Make prediction on preprocessed image"""
    # Add batch dimension
    img_batch = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(img_batch, verbose=0)[0]

    # Get results
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]

    return {
        "class": CLASS_NAMES[predicted_idx],
        "class_idx": predicted_idx,
        "confidence": float(confidence),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, predictions)}
    }


def main():
    """Main Streamlit app"""

    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a deep learning model (X-Farmer) to detect
        and classify brain tumors from MRI images.

        **Supported tumor types:**
        - Glioma
        - Meningioma
        - Pituitary tumor
        - No tumor (Normal)
        """)

        st.header("📋 Instructions")
        st.write("""
        1. Upload an MRI brain scan image
        2. Wait for the model to analyze
        3. View the prediction results

        **Supported formats:** JPG, JPEG, PNG
        """)

        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for educational purposes only.
        It should NOT be used for actual medical diagnosis.
        Always consult a qualified medical professional.
        """)

    # Load model
    model = load_model()

    if model is None:
        st.error("""
        ⚠️ **Model not found!**

        Please train the model first by running:
        ```
        python train.py
        ```
        """)
        return

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")

        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a brain MRI scan image"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    with col2:
        st.header("🔍 Analysis Results")

        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                # Preprocess
                processed_image = preprocess_image(image)

                # Predict
                result = make_prediction(model, processed_image)

            # Display results
            is_tumor = result['class'] != 'notumor'

            if is_tumor:
                st.markdown(f"""
                <div class="prediction-box tumor-detected">
                    <h2>⚠️ Tumor Detected</h2>
                    <h3>Type: {result['class'].title()}</h3>
                    <p>Confidence: {result['confidence']*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box no-tumor">
                    <h2>✅ No Tumor Detected</h2>
                    <p>Confidence: {result['confidence']*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # Probability chart
            st.subheader("📊 Class Probabilities")

            probs = result['probabilities']
            chart_data = {
                'Class': list(probs.keys()),
                'Probability': [v * 100 for v in probs.values()]
            }

            import pandas as pd
            df = pd.DataFrame(chart_data)

            st.bar_chart(df.set_index('Class'))

            # Detailed probabilities
            st.subheader("📋 Detailed Results")
            for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(prob)
                with col_b:
                    st.write(f"{cls}: {prob*100:.1f}%")

        else:
            st.info("👆 Upload an MRI image to see the analysis results")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        <p>Brain Tumor Detection using X-Farmer Model</p>
        <p>Built with Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
