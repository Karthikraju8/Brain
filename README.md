# Brain Tumor Detection using X-Farmer Model

A deep learning project for automatic detection and classification of brain tumors from MRI images using a custom X-Farmer neural network architecture.

## Project Overview

This project uses computer vision and deep learning to analyze MRI brain scans and classify them into:
- **Glioma** - Tumor arising from glial cells
- **Meningioma** - Tumor arising from the meninges
- **Pituitary** - Tumor in the pituitary gland
- **No Tumor** - Normal brain scan

## Project Structure

```
brain-tumor-detection/
├── config.py                 # Configuration and hyperparameters
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── predict.py                # Inference script
├── requirements.txt          # Dependencies
│
├── src/
│   ├── data/
│   │   ├── dataset.py        # Data loading and management
│   │   ├── preprocessing.py  # Image preprocessing
│   │   └── augmentation.py   # Data augmentation
│   │
│   ├── models/
│   │   ├── xfarmer.py        # X-Farmer model architecture
│   │   └── callbacks.py      # Training callbacks
│   │
│   └── utils/
│       ├── visualization.py  # Plotting utilities
│       ├── metrics.py        # Evaluation metrics
│       └── helpers.py        # Helper functions
│
├── app/
│   └── streamlit_app.py      # Web application
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── data/
│   ├── raw/                  # Original dataset
│   ├── processed/            # Preprocessed data
│   └── augmented/            # Augmented data
│
├── models/
│   ├── checkpoints/          # Training checkpoints
│   └── saved/                # Final saved models
│
├── results/
│   ├── plots/                # Visualization outputs
│   └── reports/              # Evaluation reports
│
└── logs/                     # Training logs
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

Download the brain tumor MRI dataset and organize it as follows:
```
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
```

**Dataset Sources:**
- [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Usage

### Training

```bash
# Basic training
python train.py

# Custom parameters
python train.py --epochs 50 --batch_size 16 --lr 0.0001

# Use lightweight model
python train.py --model xfarmer_lite
```

### Evaluation

```bash
python evaluate.py --model_path models/saved/xfarmer_final.keras
```

### Inference

```bash
# Single image
python predict.py --image path/to/mri_scan.jpg

# Multiple images
python predict.py --image_dir path/to/images/ --show
```

### Web Application

```bash
streamlit run app/streamlit_app.py
```

## X-Farmer Model Architecture

The X-Farmer (eXtended Feature Aggregation and Recognition Model for MRI) is a custom CNN designed for medical image analysis:

- **Multi-scale feature extraction** with progressive filter sizes
- **Residual connections** for better gradient flow
- **Spatial attention mechanisms** for tumor region focus
- **Batch normalization** for stable training

```
Input (224x224x3)
    ↓
Conv Block (32 filters) → Residual Block → MaxPool
    ↓
Conv Block (64 filters) → Residual Block → MaxPool
    ↓
Conv Block (128 filters) → Residual Block + Attention → MaxPool
    ↓
Conv Block (256 filters) → Residual Block + Attention → MaxPool
    ↓
Global Average Pooling
    ↓
Dense (512) → BatchNorm → ReLU → Dropout
    ↓
Dense (256) → BatchNorm → ReLU → Dropout
    ↓
Output (4 classes) → Softmax
```

## Results

| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| Precision (Macro) | ~94% |
| Recall (Macro) | ~94% |
| F1-Score (Macro) | ~94% |

*Results may vary based on dataset and training configuration.*

## Configuration

Modify `config.py` to adjust:
- Image size and preprocessing parameters
- Model architecture (filters, layers, dropout)
- Training hyperparameters (epochs, batch size, learning rate)
- Data augmentation settings

## Technologies

- **Python 3.9+**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **NumPy/Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **Streamlit** - Web application

## License

This project is for educational purposes only.

## Disclaimer

This tool is designed for educational and research purposes. It should NOT be used for actual medical diagnosis. Always consult qualified medical professionals for health-related decisions.

## Author

Your Name

## Acknowledgments

- Dataset providers on Kaggle
- TensorFlow/Keras team
- Medical imaging research community
