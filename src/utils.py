"""
Utility functions for visualization and evaluation
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def visualize_samples(x_data, y_data, num_samples=12, figsize=(12, 8)):
    """
    Visualize sample images with labels
    
    Args:
        x_data: Image data
        y_data: Labels (0=Cat, 1=Dog)
        num_samples (int): Number of samples to display
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    for i in range(min(num_samples, len(x_data))):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_data[i])
        
        label = "Cat" if y_data[i] == 0 else "Dog"
        plt.title(label)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plot training history (accuracy and loss)
    
    Args:
        history: Keras training history object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_model(model, x_test, y_test):
    """
    Evaluate model performance and display metrics
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Get predictions
    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT")
    print("-"*60)
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(y_true, y_pred, labels=['Cat', 'Dog']):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels (list): Class labels
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "-"*60)
    print("CONFUSION MATRIX STATISTICS")
    print("-"*60)
    print(f"True Negatives (Correctly predicted Cats): {tn}")
    print(f"False Positives (Cats predicted as Dogs): {fp}")
    print(f"False Negatives (Dogs predicted as Cats): {fn}")
    print(f"True Positives (Correctly predicted Dogs): {tp}")
    print(f"\nTotal Correct: {tn + tp} / {tn + fp + fn + tp}")
    print(f"Total Incorrect: {fp + fn} / {tn + fp + fn + tp}")


def display_sample_predictions(model, x_test, y_test, num_samples=8):
    """
    Display sample predictions with actual labels
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels
        num_samples (int): Number of samples to display
    """
    # Get random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 8))
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 4, i + 1)
        plt.imshow(x_test[idx])
        
        # Get prediction
        img_batch = np.expand_dims(x_test[idx], axis=0)
        pred_prob = model.predict(img_batch, verbose=0)[0][0]
        pred_label = "Dog" if pred_prob >= 0.5 else "Cat"
        actual_label = "Dog" if y_test[idx] == 1 else "Cat"
        
        # Color: green if correct, red if wrong
        color = 'green' if pred_label == actual_label else 'red'
        
        plt.title(f"Pred: {pred_label} ({pred_prob:.2f})\nActual: {actual_label}", 
                 color=color, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
