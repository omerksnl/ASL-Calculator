"""
Compare accuracies of Local, IID, and Non-IID trained models
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
DATA_DIR = Path("data/master_dataset/master_data")
MODELS_DIR = Path("models")
BATCH_SIZE = 32
IMG_SIZE = 224

# Class mapping
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'divide', 'equals', 'minus', 'multiply', 'plus']


class ASLDataset(Dataset):
    """Custom Dataset for ASL Gesture Images"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
        
        for class_name in CLASSES:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


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
    """Get preprocessing transforms"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def load_model(model_path, device):
    """Load model from checkpoint"""
    if not model_path.exists():
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    model = ASLModel(num_classes=len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, test_loader, device, model_name):
    """Evaluate model on test set"""
    if model is None:
        print(f"\n{model_name}: Model not found")
        return None
    
    print(f"\nEvaluating {model_name}...")
    model.eval()
    
    correct = 0
    total = 0
    class_correct = {cls: 0 for cls in CLASSES}
    class_total = {cls: 0 for cls in CLASSES}
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=model_name):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                class_name = CLASSES[label.item()]
                class_total[class_name] += 1
                if label == pred:
                    class_correct[class_name] += 1
    
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for cls in CLASSES:
        if class_total[cls] > 0:
            class_accuracies[cls] = 100 * class_correct[cls] / class_total[cls]
        else:
            class_accuracies[cls] = 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'class_accuracies': class_accuracies
    }


def find_model(pattern):
    """Find model matching pattern"""
    models = list(MODELS_DIR.rglob(pattern))
    if models:
        return max(models, key=lambda p: p.stat().st_mtime)  # Return latest
    return None


def main():
    print("=" * 80)
    print("Model Comparison: Local vs IID vs Non-IID")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    full_dataset = ASLDataset(DATA_DIR, transform=get_transform())
    
    # Use same split as training (80/20)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    _, test_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    print(f"✓ Test set: {len(test_dataset)} images")
    
    # Find models
    print("\nSearching for models...")
    local_path = find_model("asl_model_best_*.pth")
    iid_path = find_model("federated/federated_model_best_*.pth")
    noniid_path = None
    
    # Try to distinguish IID vs Non-IID (by timestamp or manual naming)
    fed_models = list((MODELS_DIR / "federated").glob("federated_model_best_*.pth"))
    if len(fed_models) >= 2:
        fed_models.sort(key=lambda p: p.stat().st_mtime)
        iid_path = fed_models[0]
        noniid_path = fed_models[1]
    elif len(fed_models) == 1:
        iid_path = fed_models[0]
    
    if local_path:
        print(f"✓ Local model: {local_path.name}")
    else:
        print("✗ Local model not found")
    
    if iid_path:
        print(f"✓ IID model: {iid_path.name}")
    else:
        print("✗ IID model not found")
    
    if noniid_path:
        print(f"✓ Non-IID model: {noniid_path.name}")
    else:
        print("✗ Non-IID model not found")
    
    # Load models
    local_model, local_ckpt = load_model(local_path, device) if local_path else (None, None)
    iid_model, iid_ckpt = load_model(iid_path, device) if iid_path else (None, None)
    noniid_model, noniid_ckpt = load_model(noniid_path, device) if noniid_path else (None, None)
    
    # Evaluate all models
    local_results = evaluate_model(local_model, test_loader, device, "Local Training")
    iid_results = evaluate_model(iid_model, test_loader, device, "IID Federated")
    noniid_results = evaluate_model(noniid_model, test_loader, device, "Non-IID Federated")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    print("\nOverall Accuracies:")
    print("-" * 80)
    
    if local_results:
        print(f"Local Training:      {local_results['accuracy']:.2f}% ({local_results['correct']}/{local_results['total']})")
    if iid_results:
        print(f"IID Federated:       {iid_results['accuracy']:.2f}% ({iid_results['correct']}/{iid_results['total']})")
    if noniid_results:
        print(f"Non-IID Federated:   {noniid_results['accuracy']:.2f}% ({noniid_results['correct']}/{noniid_results['total']})")
    
    # Per-class comparison
    if any([local_results, iid_results, noniid_results]):
        print("\nPer-Class Accuracies:")
        print("-" * 80)
        print(f"{'Class':<10} {'Local':<12} {'IID':<12} {'Non-IID':<12}")
        print("-" * 80)
        
        for cls in CLASSES:
            local_acc = f"{local_results['class_accuracies'][cls]:.1f}%" if local_results else "N/A"
            iid_acc = f"{iid_results['class_accuracies'][cls]:.1f}%" if iid_results else "N/A"
            noniid_acc = f"{noniid_results['class_accuracies'][cls]:.1f}%" if noniid_results else "N/A"
            print(f"{cls:<10} {local_acc:<12} {iid_acc:<12} {noniid_acc:<12}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if local_results and iid_results:
        diff = local_results['accuracy'] - iid_results['accuracy']
        print(f"\nLocal vs IID difference: {diff:+.2f}%")
        if abs(diff) < 2:
            print("→ Similar performance - IID federated learning works as well as local!")
        elif diff > 0:
            print("→ Local slightly better - expected due to more data on single device")
        else:
            print("→ IID better - federated learning benefits from data diversity!")
    
    if iid_results and noniid_results:
        diff = iid_results['accuracy'] - noniid_results['accuracy']
        print(f"\nIID vs Non-IID difference: {diff:+.2f}%")
        if diff > 5:
            print("→ Significant gap - Non-IID data distribution severely impacts FL performance")
        elif diff > 0:
            print("→ IID better - data distribution affects federated learning")
        else:
            print("→ Similar performance - model handles heterogeneous data well!")
    
    # Training info
    print("\n" + "=" * 80)
    print("TRAINING INFO")
    print("=" * 80)
    
    if local_ckpt and 'epoch' in local_ckpt:
        print(f"\nLocal model: Trained for {local_ckpt['epoch']} epochs")
    
    if iid_ckpt and 'round' in iid_ckpt:
        print(f"IID model: Trained for {iid_ckpt['round']} federated rounds")
    
    if noniid_ckpt and 'round' in noniid_ckpt:
        print(f"Non-IID model: Trained for {noniid_ckpt['round']} federated rounds")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

