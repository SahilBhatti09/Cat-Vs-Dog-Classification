"""
CNN model architecture for Cat vs Dog classification
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def create_cnn_model(input_shape=(64, 64, 3), learning_rate=0.001):
    """
    Create CNN model for binary classification (Cat vs Dog)
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        learning_rate (float): Learning rate for Adam optimizer
    
    Returns:
        keras.Model: Compiled CNN model
    """
    # Initialize sequential model
    cnn = Sequential()
    
    # Convolutional Layer 1
    cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    cnn.add(MaxPooling2D(2, 2))
    
    # Convolutional Layer 2
    cnn.add(Conv2D(64, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D(2, 2))
    
    # Flatten
    cnn.add(Flatten())
    
    # Fully Connected Layer 1
    cnn.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    cnn.add(Dropout(0.5))
    
    # Fully Connected Layer 2
    cnn.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    cnn.add(Dropout(0.5))
    
    # Output Layer
    cnn.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    cnn.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model created successfully!")
    return cnn


def get_model_summary(model):
    """
    Print model architecture summary
    
    Args:
        model: Keras model
    """
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    print("="*60 + "\n")
