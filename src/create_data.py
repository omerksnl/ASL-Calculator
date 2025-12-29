import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data/master_dataset/master_data")
CLASSES = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    '+': 'plus', '-': 'minus', '*': 'multiply', '/': 'divide', '=': 'equals'
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,  # Balanced for stability
    min_tracking_confidence=0.6,   # Balanced for smoother tracking
    model_complexity=1             # Better accuracy for complex poses
)

def create_class_folders():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES.values():
        class_dir = DATA_DIR / class_name
        class_dir.mkdir(exist_ok=True)
    print(f"✓ Data directory ready: {DATA_DIR}")

def get_hand_bbox(landmarks, image_shape):
    h, w = image_shape[:2]
    x_coords = [landmark.x * w for landmark in landmarks.landmark]
    y_coords = [landmark.y * h for landmark in landmarks.landmark]

    padding = 20
    x_min = max(0, int(min(x_coords) - padding))
    y_min = max(0, int(min(y_coords) - padding))
    x_max = min(w, int(max(x_coords) + padding))
    y_max = min(h, int(max(y_coords) + padding))
    if x_max > x_min and y_max > y_min:
        return (x_min, y_min, x_max, y_max)
    return None

def get_hand_center(landmarks, image_shape):
    """Get the center point of a hand"""
    h, w = image_shape[:2]
    x_coords = [landmark.x * w for landmark in landmarks.landmark]
    y_coords = [landmark.y * h for landmark in landmarks.landmark]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return (center_x, center_y)

def check_hand_separation(hand_landmarks_list, image_shape, min_distance=80):
    """Check if hands are well separated (not colliding)"""
    if len(hand_landmarks_list) < 2:
        return True, None
    
    centers = [get_hand_center(hand, image_shape) for hand in hand_landmarks_list]
    
    # Calculate distance between hand centers
    dx = centers[0][0] - centers[1][0]
    dy = centers[0][1] - centers[1][1]
    distance = np.sqrt(dx**2 + dy**2)
    
    return distance >= min_distance, distance

def crop_hand_region(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

def save_image(image, class_name, count):
    class_dir = DATA_DIR / CLASSES[class_name]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use class folder name instead of symbol for filename to avoid Windows filename issues
    safe_name = CLASSES[class_name]
    filename = f"{safe_name}_{count:04d}_{timestamp}.jpg"
    filepath = class_dir / filename
    resized = cv2.resize(image, (224, 224))
    success = cv2.imwrite(str(filepath), resized)
    
    if not success:
        print(f"✗ Warning: Failed to write file {filepath}")
    
    return filepath

def count_images_in_class(class_name):
    class_dir = DATA_DIR / CLASSES[class_name]
    if class_dir.exists():
        return len(list(class_dir.glob("*.jpg")))
    return 0

def main():
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

    create_class_folders()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Error: Could not open webcam")
        return

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

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            annotated_frame = frame.copy()
            bbox = None
            all_bboxes = []
            hands_well_separated = True
            hand_distance = None

            if results.multi_hand_landmarks:
                # Check hand separation first
                hands_well_separated, hand_distance = check_hand_separation(
                    results.multi_hand_landmarks, frame.shape
                )
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

                    hand_bbox = get_hand_bbox(hand_landmarks, frame.shape)
                    if hand_bbox:
                        all_bboxes.append(hand_bbox)
                        x_min, y_min, x_max, y_max = hand_bbox
                        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
                # Combine all bounding boxes into one
                if all_bboxes:
                    x_mins = [bbox[0] for bbox in all_bboxes]
                    y_mins = [bbox[1] for bbox in all_bboxes]
                    x_maxs = [bbox[2] for bbox in all_bboxes]
                    y_maxs = [bbox[3] for bbox in all_bboxes]
                    
                    bbox = (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))
                    
                    # Draw the combined bounding box - always green for 2 hands
                    if len(all_bboxes) > 1:
                        box_color = (0, 255, 0)  # Green for 2 hands detected
                        cv2.rectangle(annotated_frame, 
                                    (bbox[0], bbox[1]), 
                                    (bbox[2], bbox[3]), 
                                    box_color, 3)

            cv2.putText(annotated_frame, "Press 0-9, +, -, *, /, = to save | 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show hand detection status
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                
                if num_hands == 2:
                    if hands_well_separated:
                        hand_text = f"Hands detected: {num_hands} - Good!"
                        color = (0, 255, 0)  # Green
                    else:
                        hand_text = f"Hands detected: {num_hands} - Close"
                        color = (255, 200, 0)  # Yellow-orange warning
                else:
                    hand_text = f"Hands detected: {num_hands}"
                    color = (255, 255, 0)  # Yellow
                    
                cv2.putText(annotated_frame, hand_text, 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if current_class:
                class_name = CLASSES[current_class]
                count = image_counts[current_class]
                cv2.putText(annotated_frame, f"Current: {class_name} ({count} images)", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('ASL Data Collection', annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('c'):
                print("\nCurrent image counts:")
                for key_char, class_name in CLASSES.items():
                    count = image_counts[key_char]
                    print(f"  {class_name}: {count} images")
                print()
            elif chr(key) in CLASSES:
                class_key = chr(key)
                if bbox:
                    cropped = crop_hand_region(frame, bbox)
                    if cropped.size > 0:
                        filepath = save_image(cropped, class_key, image_counts[class_key])
                        image_counts[class_key] += 1
                        
                        # Add subtle separation quality indicator for 2-hand gestures
                        quality_msg = ""
                        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                            quality_msg = " ✓" if hands_well_separated else ""
                        
                        print(f"✓ Saved: {filepath.name} ({image_counts[class_key]} total for '{CLASSES[class_key]}'){quality_msg}")
                    else:
                        print(f"✗ Failed to crop image for '{CLASSES[class_key]}'")
                else:
                    print(f"⚠ No hand detected. Please show your hand before pressing '{class_key}'")
            elif key != 255:
                print(f"ℹ Unknown key: {chr(key)} (press 0-9, +, -, *, /, =, 'q', or 'c')")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

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

