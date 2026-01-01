"""
ASL Calculator App - Interactive calculator using gesture recognition
Automatically selects the best performing model from Local/IID/Non-IID
"""
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
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 5

# Class mapping
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'divide', 'equals', 'minus', 'multiply', 'plus']

SYMBOLS = {
    'divide': '÷',
    'equals': '=',
    'minus': '-',
    'multiply': '×',
    'plus': '+'
}

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
        self.backbone = models.resnet18(pretrained=pretrained)
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


def find_best_model():
    """Find and select the best model among Local, IID, Non-IID"""
    print("\n" + "=" * 70)
    print("Searching for trained models...")
    print("=" * 70)
    
    models_found = {}
    
    # Find Local model
    local_models = list(MODELS_DIR.glob("asl_model_best_*.pth"))
    if local_models:
        local_path = max(local_models, key=lambda p: p.stat().st_mtime)
        models_found['Local'] = local_path
        print(f"✓ Local model: {local_path.name}")
    
    # Find Federated models
    fed_models = list((MODELS_DIR / "federated").glob("federated_model_best_*.pth"))
    if len(fed_models) >= 2:
        fed_models.sort(key=lambda p: p.stat().st_mtime)
        models_found['IID'] = fed_models[0]
        models_found['Non-IID'] = fed_models[1]
        print(f"✓ IID model: {fed_models[0].name}")
        print(f"✓ Non-IID model: {fed_models[1].name}")
    elif len(fed_models) == 1:
        models_found['Federated'] = fed_models[0]
        print(f"✓ Federated model: {fed_models[0].name}")
    
    if not models_found:
        print("✗ No trained models found!")
        print("  Please train a model first:")
        print("    - Local: python src/train_local.py")
        print("    - Federated: python src/server.py + clients")
        return None, None
    
    print("\n" + "-" * 70)
    print("Available models:")
    for i, (name, path) in enumerate(models_found.items(), 1):
        print(f"  {i}. {name}")
    
    print("\nSelect model:")
    for i, name in enumerate(models_found.keys(), 1):
        print(f"  {i}) {name}")
    print(f"  0) Auto-select best")
    
    choice = input("\nChoice [0-{}]: ".format(len(models_found))).strip()
    
    if choice == '0' or not choice:
        # Auto-select: prefer Local > IID > Federated > Non-IID
        if 'Local' in models_found:
            selected = 'Local'
        elif 'IID' in models_found:
            selected = 'IID'
        elif 'Federated' in models_found:
            selected = 'Federated'
        else:
            selected = 'Non-IID'
        print(f"\n✓ Auto-selected: {selected}")
    else:
        try:
            idx = int(choice) - 1
            selected = list(models_found.keys())[idx]
        except:
            selected = list(models_found.keys())[0]
    
    model_path = models_found[selected]
    return model_path, selected


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"\nLoading model: {model_path.name}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model = ASLModel(num_classes=len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get training info
    info = {}
    if 'epoch' in checkpoint:
        info['type'] = 'Local'
        info['trained'] = f"{checkpoint['epoch']} epochs"
    elif 'round' in checkpoint:
        info['type'] = 'Federated'
        info['trained'] = f"{checkpoint['round']} rounds"
    
    if 'best_acc' in checkpoint:
        info['accuracy'] = f"{checkpoint['best_acc']:.2f}%"
    elif 'best_accuracy' in checkpoint:
        info['accuracy'] = f"{checkpoint['best_accuracy']:.2f}%"
    
    print(f"✓ Model loaded successfully!")
    if 'trained' in info:
        print(f"  Training: {info['trained']}")
    if 'accuracy' in info:
        print(f"  Best accuracy: {info['accuracy']}")
    
    return model


def get_hand_bbox(landmarks, image_shape):
    """Get bounding box for hand landmarks"""
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
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASSES[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score


def evaluate_expression(expr):
    """Safely evaluate mathematical expression"""
    # Replace symbols with operators
    expr = expr.replace('×', '*').replace('÷', '/')
    
    try:
        result = eval(expr)
        return str(result)
    except:
        return "Error"


def draw_calculator_ui(frame, expression, result, prediction, confidence, fps, num_hands, model_name):
    """Draw calculator UI on frame"""
    h, w = frame.shape[:2]
    
    # Top panel - Expression and Result
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 140), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    
    # Model info
    cv2.putText(frame, f"ASL Calculator - {model_name} Model", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Expression display
    cv2.putText(frame, "Expression:", 
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    expr_display = expression if expression else "..."
    cv2.putText(frame, expr_display, 
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    
    # Result display
    if result:
        cv2.putText(frame, f"= {result}", 
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Bottom panel - Current gesture
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - 120), (w - 10, h - 10), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    
    # Current gesture
    if prediction and confidence > CONFIDENCE_THRESHOLD:
        gesture = SYMBOLS.get(prediction, prediction)
        color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
        
        cv2.putText(frame, f"Detected: {gesture}", 
                    (20, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", 
                    (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Confidence bar
        bar_width = int((w - 40) * confidence)
        cv2.rectangle(frame, (20, h - 35), (20 + bar_width, h - 25), color, -1)
        cv2.rectangle(frame, (20, h - 35), (w - 20, h - 25), (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Show gesture to camera...", 
                    (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    
    # Instructions
    cv2.putText(frame, f"FPS: {fps:.1f} | Hands: {num_hands} | Press: Space=Add, C=Clear, Q=Quit", 
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    return frame


def main():
    print("=" * 70)
    print("ASL CALCULATOR APP")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find and load best model
    model_path, model_name = find_best_model()
    if model_path is None:
        return
    
    model = load_model(model_path, device)
    transform = get_transform()
    
    # Initialize webcam
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Webcam initialized")
    print("\n" + "=" * 70)
    print("CALCULATOR READY!")
    print("=" * 70)
    print("\nControls:")
    print("  SPACE - Add gesture to expression")
    print("  C     - Clear expression")
    print("  Q     - Quit")
    print("\nShow gestures one at a time, press SPACE to add them!")
    print("=" * 70)
    print()
    
    # State
    expression = ""
    result = ""
    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
    confidence_buffer = deque(maxlen=SMOOTHING_WINDOW)
    fps_buffer = deque(maxlen=30)
    prev_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
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
                
                if all_bboxes:
                    combined_bbox = combine_bboxes(all_bboxes)
                    x_min, y_min, x_max, y_max = combined_bbox
                    
                    # Draw bbox
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), 
                                (255, 0, 255), 2)
                    
                    # Predict
                    hand_region = frame[y_min:y_max, x_min:x_max]
                    if hand_region.size > 0:
                        pred_class, conf = predict_gesture(
                            model, hand_region, transform, device
                        )
                        
                        prediction_buffer.append(pred_class)
                        confidence_buffer.append(conf)
                        
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
            annotated_frame = draw_calculator_ui(
                annotated_frame, expression, result, 
                prediction, confidence, avg_fps, num_hands, model_name
            )
            
            cv2.imshow('ASL Calculator', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord(' '):  # Space - add gesture
                if prediction and confidence > CONFIDENCE_THRESHOLD:
                    if prediction == 'equals':
                        # Calculate result
                        if expression:
                            result = evaluate_expression(expression)
                            print(f"  {expression} = {result}")
                    else:
                        # Add to expression
                        symbol = SYMBOLS.get(prediction, prediction)
                        expression += symbol
                        result = ""
                        print(f"  Expression: {expression}")
            elif key == ord('c'):  # Clear
                expression = ""
                result = ""
                print("  Expression cleared")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        
        print("\n" + "=" * 70)
        print("Calculator closed")
        print("=" * 70)


if __name__ == "__main__":
    main()

