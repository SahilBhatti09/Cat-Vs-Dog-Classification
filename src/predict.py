"""
Prediction utilities for Cat vs Dog classification
"""

import cv2
import numpy as np
from .data_loader import load_single_image


def predict_single_image(model, image_path, target_size=(64, 64), threshold=0.5):
    """
    Predict if an image is a cat or dog
    
    Args:
        model: Trained Keras model
        image_path (str): Path to image file
        target_size (tuple): Target image size
        threshold (float): Classification threshold (default: 0.5)
    
    Returns:
        tuple: (prediction, probability, label)
            - prediction: 0 (cat) or 1 (dog)
            - probability: confidence score (0-1)
            - label: "Cat" or "Dog"
    """
    # Load and preprocess image
    img = load_single_image(image_path, target_size)
    
    # Add batch dimension
    img_batch = np.expand_dims(img, axis=0)
    
    # Make prediction
    probability = model.predict(img_batch, verbose=0)[0][0]
    
    # Convert to binary prediction
    prediction = 1 if probability >= threshold else 0
    label = "Dog" if prediction == 1 else "Cat"
    
    return prediction, probability, label


def predict_batch(model, image_paths, target_size=(64, 64), threshold=0.5):
    """
    Predict multiple images at once
    
    Args:
        model: Trained Keras model
        image_paths (list): List of image file paths
        target_size (tuple): Target image size
        threshold (float): Classification threshold
    
    Returns:
        list: List of tuples (prediction, probability, label) for each image
    """
    results = []
    
    for image_path in image_paths:
        prediction, probability, label = predict_single_image(
            model, image_path, target_size, threshold
        )
        results.append({
            'image_path': image_path,
            'prediction': prediction,
            'probability': probability,
            'label': label
        })
    
    return results


def predict_and_display(model, image_path, target_size=(64, 64)):
    """
    Predict and display result with confidence
    
    Args:
        model: Trained Keras model
        image_path (str): Path to image file
        target_size (tuple): Target image size
    """
    prediction, probability, label = predict_single_image(model, image_path, target_size)
    
    confidence = probability if prediction == 1 else (1 - probability)
    
    print(f"\nPrediction Results:")
    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw Probability: {probability:.4f}")
    
    if confidence < 0.6:
        print("⚠️  Low confidence - model is uncertain")
    elif confidence < 0.8:
        print("✓ Moderate confidence")
    else:
        print("✓✓ High confidence")


def batch_predict_and_display(model, image_paths, target_size=(64, 64)):
    """
    Predict and display results for multiple images
    
    Args:
        model: Trained Keras model
        image_paths (list): List of image file paths
        target_size (tuple): Target image size
    """
    print("\n" + "="*70)
    print("BATCH PREDICTION RESULTS")
    print("="*70)
    
    results = predict_batch(model, image_paths, target_size)
    
    cat_count = 0
    dog_count = 0
    
    for result in results:
        prediction = result['prediction']
        probability = result['probability']
        label = result['label']
        image_path = result['image_path']
        
        confidence = probability if prediction == 1 else (1 - probability)
        
        print(f"\n{image_path}")
        print(f"  → Prediction: {label} ({confidence:.2%} confidence)")
        
        if label == "Cat":
            cat_count += 1
        else:
            dog_count += 1
    
    print("\n" + "="*70)
    print(f"Summary: {cat_count} Cats, {dog_count} Dogs")
    print("="*70)


def predict_frame(model, frame, target_size=(64, 64)):
    """
    Predict cat or dog from a video frame (for webcam integration)
    
    Args:
        model: Trained Keras model
        frame: Video frame from cv2.VideoCapture (BGR format)
        target_size (tuple): Target image size for model input
    
    Returns:
        float: Raw prediction probability (0-1)
               - Values closer to 0 indicate Cat
               - Values closer to 1 indicate Dog
    
    Note:
        This function handles BGR to RGB conversion automatically,
        which is critical for webcam frames from OpenCV
    """
    # Resize to model input size
    img = cv2.resize(frame, target_size)
    
    # Convert BGR to RGB (VERY IMPORTANT for OpenCV frames)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to 0-1 range
    img = img / 255.0
    
    # Add batch dimension: (64, 64, 3) -> (1, 64, 64, 3)
    img = np.expand_dims(img, axis=0)
    
    # Predict
    prediction = model.predict(img, verbose=0)
    
    # Return raw probability
    return prediction[0][0]


def run_webcam_prediction(model, target_size=(64, 64), camera_id=0, window_name="Webcam Prediction"):
    """
    Run real-time webcam prediction with image capture functionality
    
    Args:
        model: Trained Keras model
        target_size (tuple): Target image size for model input
        camera_id (int): Camera device ID (default: 0 for built-in webcam)
        window_name (str): Name of the OpenCV window
    
    Controls:
        - Press 'c' to capture and save the current frame
        - Press 'q' to quit
    
    Returns:
        None
    
    Note:
        Captured images are saved in the current directory with timestamp
        Format: capture_YYYYMMDD_HHMMSS.jpg
    """
    import time
    
    # Initialize webcam
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam started!")
    print("Controls:")
    print("  - Press 'c' to capture and save frame")
    print("  - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Get prediction
        prob = predict_frame(model, frame, target_size)
        
        # Determine label and confidence
        label = "Dog" if prob > 0.5 else "Cat"
        confidence = prob if prob > 0.5 else 1 - prob
        
        # Display prediction on frame
        cv2.putText(
            frame,
            f"Prediction: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Capture frame on 'c' key
        if key == ord('c'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved {filename}")
        
        # Quit on 'q' key
        elif key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed")
