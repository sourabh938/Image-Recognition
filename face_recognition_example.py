"""
Face Recognition Example

This script demonstrates a basic face recognition system using the face_recognition
library and OpenCV to detect and identify faces in images or video streams.
"""

import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

def load_known_faces(faces_dir):
    """
    Load known faces from a directory.
    
    Args:
        faces_dir: Directory containing images of known faces
        
    Returns:
        known_face_encodings: List of face encodings
        known_face_names: List of corresponding names
    """
    known_face_encodings = []
    known_face_names = []
    
    # Iterate through each person's directory
    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        if os.path.isdir(person_dir):
            # Process each image of the person
            for image_name in os.listdir(person_dir):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, image_name)
                    
                    # Load image and find face encoding
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        # Use the first face found in the image
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person_name)
                        print(f"Loaded face: {person_name} from {image_name}")
    
    return known_face_encodings, known_face_names

def process_image(image_path, known_face_encodings, known_face_names):
    """
    Process a single image for face recognition.
    
    Args:
        image_path: Path to the image
        known_face_encodings: List of known face encodings
        known_face_names: List of corresponding names
        
    Returns:
        Annotated image with recognized faces
    """
    # Load image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find all faces in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    # Loop through each face found in the image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        # Log recognition event
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Recognized: {name}")
    
    return image

def process_video(video_source, known_face_encodings, known_face_names):
    """
    Process video stream for face recognition.
    
    Args:
        video_source: Camera index or video file path
        known_face_encodings: List of known face encodings
        known_face_names: List of corresponding names
    """
    # Initialize video capture
    video_capture = cv2.VideoCapture(video_source)
    
    if not video_capture.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    print("Processing video stream. Press 'q' to quit.")
    
    # Process frames until user quits
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Resize frame for faster processing (optional)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Initialize array for face names in current frame
        face_names = []
        
        # Check each face for matches
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            
            face_names.append(name)
        
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        # Display the resulting image
        cv2.imshow('Face Recognition', frame)
        
        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    # Directory containing known faces
    faces_dir = "known_faces"
    
    # Create directory if it doesn't exist
    if not os.path.exists(faces_dir):
        print(f"Creating directory for known faces: {faces_dir}")
        os.makedirs(faces_dir)
        print(f"Please add subdirectories with person names containing their face images")
        print(f"Example structure:")
        print(f"  {faces_dir}/")
        print(f"  ├── John_Doe/")
        print(f"  │   ├── image1.jpg")
        print(f"  │   └── image2.jpg")
        print(f"  └── Jane_Smith/")
        print(f"      └── profile.jpg")
        return
    
    # Load known faces
    print("Loading known faces...")
    known_face_encodings, known_face_names = load_known_faces(faces_dir)
    
    if not known_face_encodings:
        print("No faces loaded. Please add face images to the known_faces directory.")
        return
    
    print(f"Loaded {len(known_face_encodings)} faces of {len(set(known_face_names))} people.")
    
    # Choose mode
    print("\nChoose an option:")
    print("1. Process an image")
    print("2. Process webcam video")
    choice = input("Enter your choice (1/2): ")
    
    if choice == '1':
        # Process image
        image_path = input("Enter the path to the image: ")
        if os.path.exists(image_path):
            print(f"Processing image: {image_path}")
            result_image = process_image(image_path, known_face_encodings, known_face_names)
            
            # Save and display result
            result_path = f"result_{os.path.basename(image_path)}"
            cv2.imwrite(result_path, result_image)
            print(f"Result saved to: {result_path}")
            
            # Display the image
            cv2.imshow("Face Recognition Result", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Error: Image not found at {image_path}")
    
    elif choice == '2':
        # Process webcam video
        process_video(0, known_face_encodings, known_face_names)
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main() 