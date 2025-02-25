# Machine Learning Image Recognition Examples

This project demonstrates practical applications of machine learning for image recognition, including general object recognition and face recognition.

## Overview

Image recognition is a powerful application of machine learning that analyzes digital images by processing pixel values to identify patterns, objects, and features. This repository contains two examples:

1. **General Image Recognition**: Uses a pre-trained CNN to classify images into various categories
2. **Face Recognition**: Detects and identifies faces in images or video streams

## Examples

### 1. General Image Recognition

The `image_recognition_example.py` script demonstrates how to use a pre-trained MobileNetV2 model to classify images into 1000 different categories.

#### Features

- Uses the MobileNetV2 model pre-trained on ImageNet
- Classifies images into 1000 different categories
- Displays top 5 predictions with confidence scores
- Visualizes results with matplotlib

### 2. Face Recognition

The `face_recognition_example.py` script demonstrates how to build a face recognition system that can identify people in images or video streams.

#### Features

- Detects faces in images and video streams
- Recognizes known faces based on a database of face encodings
- Works with both static images and live video from webcam
- Logs recognition events with timestamps

## Requirements

- Python 3.6+
- TensorFlow 2.4+
- NumPy
- Matplotlib
- Pillow (PIL)
- OpenCV
- face_recognition

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### General Image Recognition

1. Place an image file in the project directory or specify the path to your image
2. Update the `image_path` variable in the `main()` function of `image_recognition_example.py`
3. Run the script:

```bash
python image_recognition_example.py
```

### Face Recognition

1. Create a directory structure for known faces:
   ```
   known_faces/
   ├── Person_Name1/
   │   ├── image1.jpg
   │   └── image2.jpg
   └── Person_Name2/
       └── profile.jpg
   ```

2. Run the face recognition script:
   ```bash
   python face_recognition_example.py
   ```

3. Choose whether to process a single image or use the webcam for real-time face recognition

## How It Works

### General Image Recognition

1. The script loads a pre-trained MobileNetV2 model
2. It preprocesses the input image to match the model's requirements
3. The model makes predictions on the image
4. The top 5 predictions are displayed with their confidence scores
5. A visualization shows the image alongside the predictions

### Face Recognition

1. The script loads known faces from the `known_faces` directory
2. For each face, it computes a face encoding (a 128-dimensional vector)
3. When processing new images or video frames:
   - It detects all faces in the image
   - Computes face encodings for each detected face
   - Compares these encodings with the known face encodings
   - Identifies the closest match based on face distance
   - Draws bounding boxes and labels around recognized faces

## Applications of Image Recognition

Image recognition has numerous real-world applications:

- **Face Recognition**: Identifying individuals in photos or videos, access control systems
- **Medical Imaging**: Detecting diseases from X-rays, MRIs, etc.
- **Autonomous Vehicles**: Recognizing road signs, pedestrians, and obstacles
- **Security Systems**: Monitoring for suspicious activities
- **Retail**: Analyzing customer behavior and product placement
- **Agriculture**: Monitoring crop health and detecting pests

## Extending These Examples

You can extend these examples by:

1. Using different pre-trained models (ResNet, VGG, etc.)
2. Fine-tuning the models on your own datasets
3. Implementing real-time object detection
4. Creating a web interface for uploading and analyzing images
5. Adding notification systems when specific objects or people are recognized 