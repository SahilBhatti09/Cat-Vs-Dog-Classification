"""
Training utilities for Cat vs Dog CNN model
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def train_model(model, x_train, y_train, x_test, y_test, 
                epochs=25, batch_size=32, use_callbacks=False,
                model_save_path='models/cat_dog_model.h5'):
    """
    Train the CNN model
    
    Args:
        model: Keras model to train
        x_train: Training images
        y_train: Training labels
        x_test: Testing images
        y_test: Testing labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        use_callbacks (bool): Whether to use training callbacks
        model_save_path (str): Path to save best model
    
    Returns:
        history: Training history object
    """
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_test)}")
    print("-" * 60)
    
    # Define callbacks if requested
    callbacks = []
    if use_callbacks:
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        callbacks = [early_stop, reduce_lr, checkpoint]
        print("Using callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint")
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks if use_callbacks else None,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)
    
    # Print final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    return history


def save_model(model, filepath='models/cat_dog_model.h5'):
    """
    Save trained model to file
    
    Args:
        model: Keras model to save
        filepath (str): Path to save the model
    """
    model.save(filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath='models/cat_dog_model.h5'):
    """
    Load a saved model
    
    Args:
        filepath (str): Path to saved model
    
    Returns:
        keras.Model: Loaded model
    """
    model = tf.keras.models.load_model(filepath)
    print(f"✓ Model loaded from {filepath}")
    return model
