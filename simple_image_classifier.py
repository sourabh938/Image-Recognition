"""
Simple Image Classifier

This script demonstrates a basic image classification system using PIL and NumPy.
It analyzes an image's color distribution and basic features.
"""

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

def load_image(image_path):
    """
    Load an image from path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object
    """
    try:
        img = Image.open(image_path)
        print(f"Successfully loaded image: {image_path}")
        print(f"Image size: {img.size}")
        print(f"Image format: {img.format}")
        print(f"Image mode: {img.mode}")
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def analyze_colors(img):
    """
    Analyze the color distribution of an image.
    
    Args:
        img: PIL Image object
        
    Returns:
        Dictionary with color analysis results
    """
    # Convert image to RGB if it's not
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Get image data as numpy array
    img_array = np.array(img)
    
    # Calculate average RGB values
    avg_red = np.mean(img_array[:, :, 0])
    avg_green = np.mean(img_array[:, :, 1])
    avg_blue = np.mean(img_array[:, :, 2])
    
    # Calculate brightness
    brightness = (avg_red + avg_green + avg_blue) / 3
    
    # Determine dominant colors (simplified)
    pixels = img.resize((50, 50)).getdata()
    color_counts = Counter(pixels)
    dominant_colors = color_counts.most_common(5)
    
    return {
        "avg_red": avg_red,
        "avg_green": avg_green,
        "avg_blue": avg_blue,
        "brightness": brightness,
        "dominant_colors": dominant_colors
    }

def classify_image(analysis):
    """
    Classify the image based on color analysis.
    
    Args:
        analysis: Dictionary with color analysis results
        
    Returns:
        List of classifications with confidence scores
    """
    classifications = []
    
    # Classify based on brightness
    brightness = analysis["brightness"]
    if brightness < 50:
        classifications.append(("Dark image", 0.8))
    elif brightness > 200:
        classifications.append(("Bright image", 0.8))
    else:
        classifications.append(("Medium brightness", 0.6))
    
    # Classify based on color dominance
    avg_red = analysis["avg_red"]
    avg_green = analysis["avg_green"]
    avg_blue = analysis["avg_blue"]
    
    # Red-dominant
    if avg_red > avg_green + 20 and avg_red > avg_blue + 20:
        classifications.append(("Red-dominant image", 0.7))
    # Green-dominant
    elif avg_green > avg_red + 20 and avg_green > avg_blue + 20:
        classifications.append(("Green-dominant image", 0.7))
    # Blue-dominant
    elif avg_blue > avg_red + 20 and avg_blue > avg_green + 20:
        classifications.append(("Blue-dominant image", 0.7))
    # Grayscale-like
    elif abs(avg_red - avg_green) < 10 and abs(avg_red - avg_blue) < 10 and abs(avg_green - avg_blue) < 10:
        classifications.append(("Grayscale-like image", 0.7))
    
    return classifications

def display_results(img, analysis, classifications):
    """
    Display the image and analysis results.
    
    Args:
        img: PIL Image object
        analysis: Dictionary with color analysis results
        classifications: List of classifications with confidence scores
    """
    plt.figure(figsize=(12, 8))
    
    # Display image
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    
    # Display color distribution
    plt.subplot(2, 2, 2)
    colors = ["red", "green", "blue"]
    values = [analysis["avg_red"], analysis["avg_green"], analysis["avg_blue"]]
    plt.bar(colors, values, color=colors)
    plt.title("RGB Distribution")
    plt.ylim(0, 255)
    
    # Display classifications
    plt.subplot(2, 2, 3)
    class_names = [c[0] for c in classifications]
    confidences = [c[1] for c in classifications]
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, confidences, align="center")
    plt.yticks(y_pos, class_names)
    plt.xlabel("Confidence")
    plt.title("Classifications")
    
    # Display brightness
    plt.subplot(2, 2, 4)
    plt.pie([analysis["brightness"], 255 - analysis["brightness"]], 
            labels=["Brightness", ""],
            colors=["yellow", "gray"],
            autopct="%1.1f%%")
    plt.title(f"Brightness: {analysis['brightness']:.1f}/255")
    
    plt.tight_layout()
    plt.show()

def main():
    print("Simple Image Classifier")
    print("=======================")
    
    # Get image path from user
    default_image = "sample_image.jpg"
    image_path = input(f"Enter image path (default: {default_image}): ") or default_image
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        print("Please provide a sample image in the current directory or specify a valid path.")
        return
    
    # Load and analyze image
    img = load_image(image_path)
    if img is None:
        return
    
    print("\nAnalyzing image...")
    analysis = analyze_colors(img)
    
    print("\nClassifying image...")
    classifications = classify_image(analysis)
    
    # Display results
    print("\nResults:")
    print(f"Average RGB: R={analysis['avg_red']:.1f}, G={analysis['avg_green']:.1f}, B={analysis['avg_blue']:.1f}")
    print(f"Brightness: {analysis['brightness']:.1f}/255")
    print("\nClassifications:")
    for name, confidence in classifications:
        print(f"- {name} (confidence: {confidence:.2f})")
    
    # Ask if user wants to see visualization
    show_viz = input("\nShow visualization? (y/n): ").lower() == 'y'
    if show_viz:
        display_results(img, analysis, classifications)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 