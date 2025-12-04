"""
create_data.py - Week 2
Data Collection Script for ASL Gesture Recognition

This script uses OpenCV to capture webcam feed and MediaPipe to detect hands.
Upon pressing a number key (0-9) or operator key (+, -, *, /, =), it saves
a cropped image of the detected hand to the corresponding class folder.

Usage:
    python src/create_data.py
    
Controls:
    - 0-9: Save image to corresponding digit class folder
    - +, -, *, /, =: Save image to corresponding operator class folder
    - 'q': Quit the application
    - 'c': Change current class (for organizing data collection)
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# Configuration
DATA_DIR = Path("data/master_dataset/master_data")
CLASSES = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    '+': 'plus', '-': 'minus', '*': 'multiply', '/': 'divide', '=': 'equals'
}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def create_class_folders():
    """Create folder structure for all gesture classes."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES.values():
        class_dir = DATA_DIR / class_name
        class_dir.mkdir(exist_ok=True)
    print(f"✓ Data directory ready: {DATA_DIR}")

def get_hand_bbox(landmarks, image_shape):
    """
    Extract bounding box coordinates from hand landmarks.
    
    Args:
        landmarks: MediaPipe hand landmarks
        image_shape: Tuple of (height, width)
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) or None if invalid
    """
    h, w = image_shape[:2]
    
    # Get all landmark coordinates
    x_coords = [landmark.x * w for landmark in landmarks.landmark]
    y_coords = [landmark.y * h for landmark in landmarks.landmark]
    
    # Calculate bounding box with padding
    padding = 20  # pixels
    x_min = max(0, int(min(x_coords) - padding))
    y_min = max(0, int(min(y_coords) - padding))
    x_max = min(w, int(max(x_coords) + padding))
    y_max = min(h, int(max(y_coords) + padding))
    
    # Ensure valid bounding box
    if x_max > x_min and y_max > y_min:
        return (x_min, y_min, x_max, y_max)
    return None

def crop_hand_region(image, bbox):
    """Crop the hand region from the image."""
    x_min, y_min, x_max, y_max = bbox
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

def save_image(image, class_name, count):
    """Save cropped hand image to the appropriate class folder."""
    class_dir = DATA_DIR / CLASSES[class_name]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{class_name}_{count:04d}_{timestamp}.jpg"
    filepath = class_dir / filename
    
    # Resize to 224x224 for model input (optional, but good for consistency)
    resized = cv2.resize(image, (224, 224))
    
    cv2.imwrite(str(filepath), resized)
    return filepath

def count_images_in_class(class_name):
    """Count existing images in a class folder."""
    class_dir = DATA_DIR / CLASSES[class_name]
    if class_dir.exists():
        return len(list(class_dir.glob("*.jpg")))
    return 0

def main():
    """Main data collection loop."""
    print("=" * 60)
    print("ASL Gesture Data Collection Tool")
    print("=" * 60)
    print("\nControls:")
    print("  - Press 0-9 to save digit gestures")
    print("  - Press +, -, *, /, = to save operator gestures")
    print("  - Press 'q' to quit")
    print("  - Press 'c' to see current class counts")
    print("\nMake sure your hand is visible in the camera!")
    print("=" * 60)
    print()
    
    # Create folder structure
    create_class_folders()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Error: Could not open webcam")
        return
    
    # Set webcam resolution (optional, for better quality)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Webcam initialized")
    print("Starting capture...\n")
    
    current_class = None
    image_counts = {key: count_images_in_class(key) for key in CLASSES.keys()}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw hand landmarks and bounding box
            annotated_frame = frame.copy()
            bbox = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    # Get bounding box
                    bbox = get_hand_bbox(hand_landmarks, frame.shape)
                    
                    if bbox:
                        x_min, y_min, x_max, y_max = bbox
                        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # Display instructions
            cv2.putText(annotated_frame, "Press 0-9, +, -, *, /, = to save | 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if current_class:
                class_name = CLASSES[current_class]
                count = image_counts[current_class]
                cv2.putText(annotated_frame, f"Current: {class_name} ({count} images)", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow('ASL Data Collection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('c'):
                # Show class counts
                print("\nCurrent image counts:")
                for key_char, class_name in CLASSES.items():
                    count = image_counts[key_char]
                    print(f"  {class_name}: {count} images")
                print()
            elif chr(key) in CLASSES:
                class_key = chr(key)
                if bbox:
                    # Crop and save image
                    cropped = crop_hand_region(frame, bbox)
                    if cropped.size > 0:
                        filepath = save_image(cropped, class_key, image_counts[class_key])
                        image_counts[class_key] += 1
                        print(f"✓ Saved: {filepath.name} ({image_counts[class_key]} total for '{CLASSES[class_key]}')")
                    else:
                        print(f"✗ Failed to crop image for '{CLASSES[class_key]}'")
                else:
                    print(f"⚠ No hand detected. Please show your hand before pressing '{class_key}'")
            elif key != 255:  # Ignore no-key-pressed
                print(f"ℹ Unknown key: {chr(key)} (press 0-9, +, -, *, /, =, 'q', or 'c')")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        
        # Final summary
        print("\n" + "=" * 60)
        print("Data Collection Summary")
        print("=" * 60)
        total_images = sum(image_counts.values())
        print(f"Total images collected: {total_images}")
        for key_char, class_name in CLASSES.items():
            count = image_counts[key_char]
            print(f"  {class_name}: {count} images")
        print(f"\nData saved to: {DATA_DIR}")
        print("=" * 60)

if __name__ == "__main__":
    main()

