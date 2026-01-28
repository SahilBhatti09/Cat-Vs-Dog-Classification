# Cat vs Dog Image Classification using CNN

A Convolutional Neural Network (CNN) project for classifying images as either cats or dogs using TensorFlow/Keras. This is a beginner-friendly deep learning project demonstrating image classification with over 21,000 training images.

## Overview

This project implements a binary image classification system using a CNN to distinguish between cat and dog images. The model is trained on a dataset of 21,616 images (10,832 cats and 10,784 dogs) and achieves high accuracy in classification.

## Features

- **Data Loading & Preprocessing**: Automated loading from organized dataset folders
- **Train-Test Split**: 80-20 split with stratification for balanced classes
- **CNN Architecture**: Multi-layer convolutional neural network with regularization
- **Image Preprocessing**: Resizing (64x64) and normalization (0-1 range)
- **Visualization**: Sample image display and training metrics
- **Model Persistence**: Save and load trained models
- **Prediction Function**: Easy-to-use prediction for new images
- **Real-time Webcam Prediction**: Live cat/dog classification using webcam feed
- **Dropout & L2 Regularization**: Prevents overfitting
- **Training Visualization**: Loss and accuracy plots

## Project Structure

```
catVsDog-cnn/
│
├── catVsdog.ipynb           # Main Jupyter notebook with complete implementation
├── webCam.ipynb             # Real-time webcam prediction implementation
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
│
├── dataset/                # Training dataset
│   ├── cat/               # 10,832 cat images
│   └── dog/               # 10,784 dog images
│
├── test cases/            # Sample test images for prediction
│   ├── cat images
│   └── dog images
│
├── models/                # Saved trained models
│   └── cnn_model.keras    # Trained CNN model
│
└── src/                   # Python modules (reusable code)
    ├── __init__.py        # Package initialization
    ├── data_loader.py     # Data loading and preprocessing
    ├── model.py           # CNN model architecture
    ├── train.py           # Training logic
    ├── predict.py         # Prediction functions (including webcam)
    └── utils.py           # Visualization and utilities
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SahilBhatti09/Car-Vs-Dog-Classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

### Structure
- **Total Images**: 21,616
- **Cat Images**: 10,832 (Label: 0)
- **Dog Images**: 10,784 (Label: 1)
- **Image Format**: JPG
- **Preprocessed Size**: 64x64x3 (RGB)

### Data Split
- **Training**: 80% (~17,293 images)
- **Testing**: 20% (~4,323 images)
- **Subset for Quick Training**: 2,000 images (configurable)

## Model Architecture

```
Input Layer: (64, 64, 3)
    ↓
Conv2D(32, 3x3) + ReLU + MaxPooling(2x2)
    ↓
Conv2D(64, 3x3) + ReLU + MaxPooling(2x2)
    ↓
Flatten
    ↓
Dense(64, ReLU) + L2(0.01) + Dropout(0.5)
    ↓
Dense(32, ReLU) + L2(0.01) + Dropout(0.5)
    ↓
Dense(1, Sigmoid)
```

### Model Parameters
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Regularization**: L2 (0.01) + Dropout (0.5)
- **Total Parameters**: ~150K

## Usage

### Using the Jupyter Notebook

1. Open the notebook:
```bash
jupyter notebook catVsdog.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess data
   - Build the CNN model
   - Train the model
   - Evaluate performance
   - Make predictions on new images

### Using Python Modules (src/)

```python
from src.data_loader import load_data, preprocess_images
from src.model import create_cnn_model
from src.train import train_model
from src.predict import predict_single_image

# Load data
x_train, x_test, y_train, y_test = load_data('dataset/')

# Create model
model = create_cnn_model(input_shape=(64, 64, 3))

# Train model
history = train_model(model, x_train, y_train, x_test, y_test)

# Predict
result = predict_single_image(model, 'test cases/cat1.jpg')
print(f"Prediction: {result}")
```

## Training

### Training Configuration
- **Epochs**: 25 (configurable)
- **Batch Size**: 32
- **Validation Split**: 20%
- **Early Stopping**: Patience of 5 epochs
- **Learning Rate**: 0.001

### Training Results
- Training accuracy typically reaches ~95%
- Validation accuracy typically reaches ~85-90%
- Training time: ~15-20 minutes (with 2000 images subset)

## Model Performance

After training, the model achieves:
- **Test Accuracy**: ~85-90%
- **Training Accuracy**: ~95%
- **Loss**: Binary crossentropy ~0.2-0.3

Performance metrics include:
- Confusion Matrix
- Classification Report
- Accuracy/Loss curves

## Making Predictions

### On Test Images

```python
from src.predict import predict_single_image

# Predict on a new image
prediction = predict_single_image(model, 'path/to/image.jpg')

if prediction == 0:
    print("Prediction: Cat")
else:
    print("Prediction: Dog")
```

### Batch Predictions

```python
import os
from src.predict import predict_batch

test_images = ['test cases/cat1.jpg', 'test cases/dog1.jpg']
results = predict_batch(model, test_images)
```

### Real-time Webcam Prediction

The project includes real-time webcam prediction functionality:

**Using the Webcam Notebook:**

1. Open the webcam notebook:
```bash
jupyter notebook webCam.ipynb
```

2. Run all cells to start the webcam and see real-time predictions
3. Press 'q' to quit the webcam feed

**Note:** 
- Make sure you have opencv-python installed (already in requirements.txt)
- The webcam prediction uses the same preprocessing as training (BGR→RGB conversion, normalization)
- Works with both built-in and external webcams

## Code Structure

### Jupyter Notebooks

**catVsdog.ipynb** - Main training notebook with:
1. Data loading and preprocessing
2. Model creation and training
3. Evaluation and visualization
4. Prediction functions
5. Custom `load_image()` and `predict_image()` functions

**webCam.ipynb** - Real-time prediction notebook with:
1. Webcam integration using OpenCV
2. Live frame processing and prediction
3. Real-time display with prediction labels
4. `predict_frame()` function for video frames

### Python Modules (src/)
Reusable code organized into modules:
- `data_loader.py`: Data handling and image preprocessing
- `model.py`: CNN model architecture definition
- `train.py`: Training pipeline with callbacks
- `predict.py`: Prediction utilities (images and webcam frames)
- `utils.py`: Visualization and evaluation helpers

## Potential Enhancements

See the end of this README for suggestions on how to improve the model and project further.

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `subset_size` in data loading
   - Decrease batch size
   - Use smaller image dimensions

2. **Low Accuracy**
   - Increase number of training images
   - Train for more epochs
   - Adjust learning rate
   - Add data augmentation

3. **Overfitting**
   - Increase dropout rate
   - Add more L2 regularization
   - Use data augmentation
   - Reduce model complexity

## Test Cases

Model worked fine with 90% of test case images.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author
Sahil Bhatti

## Acknowledgments
- Dataset: Cat and Dog images dataset
- Framework: TensorFlow/Keras
- Inspiration: Classic image classification problem

---