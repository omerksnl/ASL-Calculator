import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
import numpy as np
from pathlib import Path
from collections import deque
import time

# Configuration
MODELS_DIR = Path("models")
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_WINDOW = 5  # Number of predictions to average for stability

# Class mapping
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'divide', 'equals', 'minus', 'multiply', 'plus']

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
    model_complexity=1
)


class ASLModel(nn.Module):
    """CNN Model for ASL Gesture Recognition"""
    
    def __init__(self, num_classes=15, pretrained=False):
        super(ASLModel, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_transform():
    """Get preprocessing transforms for inference"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = ASLModel(num_classes=len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    best_acc = checkpoint.get('best_acc', 0)
    epoch = checkpoint.get('epoch', 0)
    
    print(f"✓ Model loaded successfully!")
    print(f"  Epoch: {epoch}")
    print(f"  Best Validation Accuracy: {best_acc:.2f}%")
    
    return model


def get_hand_bbox(landmarks, image_shape):
    """Get bounding box for hand landmarks"""
    h, w = image_shape[:2]
    x_coords = [landmark.x * w for landmark in landmarks.landmark]
    y_coords = [landmark.y * h for landmark in landmarks.landmark]
    
    padding = 40
    x_min = max(0, int(min(x_coords) - padding))
    y_min = max(0, int(min(y_coords) - padding))
    x_max = min(w, int(max(x_coords) + padding))
    y_max = min(h, int(max(y_coords) + padding))
    
    if x_max > x_min and y_max > y_min:
        return (x_min, y_min, x_max, y_max)
    return None


def combine_bboxes(bboxes):
    """Combine multiple bounding boxes into one"""
    if not bboxes:
        return None
    if len(bboxes) == 1:
        return bboxes[0]
    
    x_mins = [bbox[0] for bbox in bboxes]
    y_mins = [bbox[1] for bbox in bboxes]
    x_maxs = [bbox[2] for bbox in bboxes]
    y_maxs = [bbox[3] for bbox in bboxes]
    
    return (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))


def predict_gesture(model, image, transform, device):
    """Make prediction on image"""
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASSES[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score, probabilities[0].cpu().numpy()


def draw_ui(frame, prediction, confidence, fps, num_hands, top_predictions=None):
    """Draw UI elements on frame"""
    h, w = frame.shape[:2]
    
    # Create semi-transparent overlay for info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Title
    cv2.putText(frame, "ASL Gesture Recognition - Live Demo", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # FPS and hands detected
    cv2.putText(frame, f"FPS: {fps:.1f} | Hands: {num_hands}", 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Prediction
    if prediction and confidence > CONFIDENCE_THRESHOLD:
        # Format prediction display
        pred_display = prediction
        if prediction in ['divide', 'equals', 'minus', 'multiply', 'plus']:
            symbols = {'divide': '÷', 'equals': '=', 'minus': '-', 'multiply': '×', 'plus': '+'}
            pred_display = f"{prediction} ({symbols[prediction]})"
        
        # Color based on confidence
        if confidence > 0.9:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.7:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 165, 255)  # Orange for low confidence
        
        cv2.putText(frame, f"Prediction: {pred_display}", 
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", 
                    (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw confidence bar
        bar_width = int((w - 40) * confidence)
        cv2.rectangle(frame, (20, 150), (20 + bar_width, 165), color, -1)
        cv2.rectangle(frame, (20, 150), (w - 20, 165), (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Prediction: Detecting...", 
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    
    # Instructions at bottom
    cv2.putText(frame, "Press 'q' to quit | 's' to screenshot | 'h' to toggle help", 
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def draw_help(frame):
    """Draw help overlay"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Semi-transparent background
    cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    
    # Help text
    help_lines = [
        "ASL Gesture Recognition Help",
        "",
        "Supported Gestures:",
        "  - Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9",
        "  - Operators: + - × ÷ =",
        "",
        "Tips:",
        "  - Keep hand(s) centered in frame",
        "  - Use good lighting",
        "  - For operators, use both hands",
        "  - Keep hands steady for best results",
        "",
        "Controls:",
        "  q - Quit",
        "  s - Save screenshot",
        "  h - Toggle this help",
        "",
        "Press 'h' to close"
    ]
    
    y_start = h//4 + 40
    for i, line in enumerate(help_lines):
        y = y_start + i * 25
        if line.startswith("ASL"):
            cv2.putText(frame, line, (w//4 + 30, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, line, (w//4 + 30, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def find_latest_model(models_dir):
    """Find the most recent best model"""
    model_files = list(models_dir.glob("asl_model_best_*.pth"))
    
    if not model_files:
        print("✗ No trained model found!")
        print(f"  Please train a model first using: python src/train_local.py")
        return None
    
    # Sort by modification time, get latest
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    return latest_model


def main():
    print("=" * 80)
    print("ASL Gesture Recognition - Live Demo")
    print("=" * 80)
    print()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Find and load model
    model_path = find_latest_model(MODELS_DIR)
    if model_path is None:
        return
    
    model = load_model(model_path, device)
    transform = get_transform()
    print()
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Webcam initialized")
    print()
    print("=" * 80)
    print("Starting live detection...")
    print("Press 'q' to quit, 's' to screenshot, 'h' for help")
    print("=" * 80)
    print()
    
    # Smoothing buffer
    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
    confidence_buffer = deque(maxlen=SMOOTHING_WINDOW)
    
    # FPS calculation
    fps_buffer = deque(maxlen=30)
    prev_time = time.time()
    
    show_help = False
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Error: Could not read frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            annotated_frame = frame.copy()
            prediction = None
            confidence = 0.0
            num_hands = 0
            
            # Detect hands and make prediction
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                all_bboxes = []
                
                # Draw hand landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    bbox = get_hand_bbox(hand_landmarks, frame.shape)
                    if bbox:
                        all_bboxes.append(bbox)
                        # Draw individual hand boxes
                        x_min, y_min, x_max, y_max = bbox
                        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), 
                                    (255, 0, 0), 2)
                
                # Combine bboxes and make prediction
                if all_bboxes:
                    combined_bbox = combine_bboxes(all_bboxes)
                    x_min, y_min, x_max, y_max = combined_bbox
                    
                    # Draw combined bbox
                    if len(all_bboxes) > 1:
                        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), 
                                    (0, 255, 0), 3)
                    
                    # Crop and predict
                    hand_region = frame[y_min:y_max, x_min:x_max]
                    if hand_region.size > 0:
                        pred_class, conf, probs = predict_gesture(
                            model, hand_region, transform, device
                        )
                        
                        # Smooth predictions
                        prediction_buffer.append(pred_class)
                        confidence_buffer.append(conf)
                        
                        # Use most common prediction in buffer
                        if len(prediction_buffer) >= SMOOTHING_WINDOW // 2:
                            prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                            confidence = np.mean(confidence_buffer)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            fps_buffer.append(fps)
            avg_fps = np.mean(fps_buffer)
            prev_time = curr_time
            
            # Draw UI
            annotated_frame = draw_ui(annotated_frame, prediction, confidence, 
                                     avg_fps, num_hands)
            
            # Show help overlay if requested
            if show_help:
                annotated_frame = draw_help(annotated_frame)
            
            # Display
            cv2.imshow('ASL Live Demo', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord('s'):
                screenshot_path = f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"✓ Screenshot saved: {screenshot_path}")
                screenshot_count += 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        
        print("\n" + "=" * 80)
        print("Demo ended")
        print("=" * 80)


if __name__ == "__main__":
    main()

