"""
Data loading and preprocessing utilities for Cat vs Dog classification
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_data(dataset_path='dataset/', test_size=0.2, subset_size=2000, random_state=42):
    """
    Load cat and dog images from dataset folder
    
    Args:
        dataset_path (str): Path to dataset folder containing 'cat' and 'dog' subfolders
        test_size (float): Fraction of data to use for testing (default: 0.2)
        subset_size (int): Number of images to use (None for all images)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    print("Loading images from dataset...")
    
    # Set paths
    cat_folder = os.path.join(dataset_path, 'cat')
    dog_folder = os.path.join(dataset_path, 'dog')
    
    # Get all image file paths
    cat_images = []
    for filename in os.listdir(cat_folder):
        if filename.endswith('.jpg'):
            cat_images.append(os.path.join(cat_folder, filename))
    
    dog_images = []
    for filename in os.listdir(dog_folder):
        if filename.endswith('.jpg'):
            dog_images.append(os.path.join(dog_folder, filename))
    
    print(f"Found {len(cat_images)} cat images")
    print(f"Found {len(dog_images)} dog images")
    
    # Create labels (0 = Cat, 1 = Dog)
    cat_labels = [0] * len(cat_images)
    dog_labels = [1] * len(dog_images)
    
    # Combine images and labels
    all_images = cat_images + dog_images
    all_labels = cat_labels + dog_labels
    
    # Split into train and test
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images,
        all_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=all_labels
    )
    
    print(f"Training images: {len(train_images)}")
    print(f"Testing images: {len(test_images)}")
    
    # Use subset if specified
    if subset_size is not None:
        train_images = train_images[:int(subset_size * 0.8)]
        train_labels = train_labels[:int(subset_size * 0.8)]
        test_images = test_images[:int(subset_size * 0.2)]
        test_labels = test_labels[:int(subset_size * 0.2)]
        print(f"\nUsing subset of {subset_size} images")
    
    # Load and preprocess images
    print("\nLoading training images...")
    x_train = preprocess_images(train_images)
    y_train = np.array(train_labels)
    
    print("Loading testing images...")
    x_test = preprocess_images(test_images)
    y_test = np.array(test_labels)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return x_train, x_test, y_train, y_test


def preprocess_images(image_paths, target_size=(64, 64)):
    """
    Load and preprocess multiple images
    
    Args:
        image_paths (list): List of image file paths
        target_size (tuple): Target image size (width, height)
    
    Returns:
        numpy.ndarray: Preprocessed images array
    """
    images = []
    for image_path in image_paths:
        img = load_single_image(image_path, target_size)
        images.append(img)
    
    return np.array(images)


def load_single_image(image_path, target_size=(64, 64)):
    """
    Load and preprocess a single image
    
    Args:
        image_path (str): Path to image file
        target_size (tuple): Target image size (width, height)
    
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Load image and resize
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Normalize: scale pixel values from 0-255 to 0-1
    img_array = img_array / 255.0
    
    return img_array
