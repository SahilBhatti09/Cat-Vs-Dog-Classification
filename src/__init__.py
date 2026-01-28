"""
Cat vs Dog CNN Classifier
A deep learning project for binary image classification

Author: Sahil Bhatti
"""

__version__ = "1.0.0"

from .data_loader import load_data, preprocess_images, load_single_image
from .model import create_cnn_model, get_model_summary
from .train import train_model
from .predict import predict_single_image, predict_batch
from .utils import visualize_samples, plot_training_history, evaluate_model

__all__ = [
    'load_data',
    'preprocess_images',
    'load_single_image',
    'create_cnn_model',
    'get_model_summary',
    'train_model',
    'predict_single_image',
    'predict_batch',
    'visualize_samples',
    'plot_training_history',
    'evaluate_model'
]
