"""
Image Recognition Example using TensorFlow and Keras

This script demonstrates a basic image recognition system using a pre-trained
convolutional neural network (CNN) model to classify images.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def load_and_prepare_image(img_path, target_size=(224, 224)):
    """
    Load an image from path and prepare it for the model.
    
    Args:
        img_path: Path to the image file
        target_size: Size to resize the image to (default: 224x224)
        
    Returns:
        Preprocessed image ready for prediction
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array), img

def predict_image(model, img_array):
    """
    Make predictions on the image using the model.
    
    Args:
        model: Pre-trained model
        img_array: Preprocessed image array
        
    Returns:
        List of predictions with class names and probabilities
    """
    predictions = model.predict(img_array)
    return decode_predictions(predictions, top=5)[0]

def display_prediction(img, predictions):
    """
    Display the image and its predictions.
    
    Args:
        img: Original image
        predictions: Model predictions
    """
    plt.figure(figsize=(10, 5))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input Image')
    
    # Display predictions
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(predictions))
    class_names = [pred[1] for pred in predictions]
    confidence = [pred[2] for pred in predictions]
    
    plt.barh(y_pos, confidence, align='center')
    plt.yticks(y_pos, class_names)
    plt.xlabel('Confidence')
    plt.title('Top Predictions')
    
    plt.tight_layout()
    plt.show()

def main():
    # Example usage
    print("Loading pre-trained MobileNetV2 model...")
    model = MobileNetV2(weights='imagenet')
    
    # Replace with your own image path
    image_path = "sample_image.jpg"
    
    try:
        # Process image
        print(f"Processing image: {image_path}")
        img_array, img = load_and_prepare_image(image_path)
        
        # Make prediction
        print("Making predictions...")
        predictions = predict_image(model, img_array)
        
        # Display results
        print("\nTop predictions:")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            print(f"{i+1}: {label} ({score:.2f})")
        
        # Visualize
        display_prediction(img, predictions)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this script:")
        print("1. Install required packages: pip install tensorflow numpy matplotlib pillow")
        print("2. Provide a valid image path")

if __name__ == "__main__":
    main() 