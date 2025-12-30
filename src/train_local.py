import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
DATA_DIR = Path("data/master_dataset/master_data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs")

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
VAL_SPLIT = 0.2
IMG_SIZE = 224
NUM_WORKERS = 4

# Training options
RESUME_TRAINING = True  # Set to True to resume from latest checkpoint
START_FROM_EPOCH = 0  # Will be overridden if resuming

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
        
        # Load all image paths and labels
        for class_name in CLASSES:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"✓ Loaded {len(self.samples)} images from {len(CLASSES)} classes")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Get the distribution of samples per class"""
        distribution = {cls: 0 for cls in CLASSES}
        for _, label in self.samples:
            class_name = CLASSES[label]
            distribution[class_name] += 1
        return distribution


def get_transforms(train=True):
    """Get data augmentation transforms"""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


class ASLModel(nn.Module):
    """CNN Model for ASL Gesture Recognition"""
    
    def __init__(self, num_classes=15, pretrained=True):
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


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels


def plot_training_history(history, save_path):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def save_checkpoint(model, optimizer, epoch, history, best_acc, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_acc': best_acc,
        'classes': CLASSES,
        'num_classes': len(CLASSES)
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved to {save_path}")


def find_latest_checkpoint(models_dir):
    """Find the most recent best model checkpoint"""
    model_files = list(models_dir.glob("asl_model_best_*.pth"))
    
    if not model_files:
        return None
    
    # Sort by modification time, get latest
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    return latest_model


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load model checkpoint and return epoch, history, best_acc"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    history = checkpoint.get('history', {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    })
    best_acc = checkpoint.get('best_acc', 0)
    
    print(f"✓ Checkpoint loaded successfully!")
    print(f"  Starting from epoch: {epoch}")
    print(f"  Best accuracy so far: {best_acc:.2f}%")
    
    return epoch, history, best_acc


def main():
    print("=" * 80)
    print("ASL Gesture Recognition - Local Training")
    print("=" * 80)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Resume training: {'YES' if RESUME_TRAINING else 'NO'}")
    print(f"Target epochs: {NUM_EPOCHS}")
    print()
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = ASLDataset(DATA_DIR, transform=None)
    
    # Show class distribution
    distribution = full_dataset.get_class_distribution()
    print("\nClass distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name:8s}: {count:4d} images")
    print(f"  {'Total':8s}: {sum(distribution.values()):4d} images")
    print()
    
    # Split dataset
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = get_transforms(train=True)
    val_dataset.dataset.transform = get_transforms(train=False)
    
    print(f"✓ Train set: {len(train_dataset)} images")
    print(f"✓ Val set: {len(val_dataset)} images")
    print()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print()
    
    # Create model
    print("Creating model...")
    model = ASLModel(num_classes=len(CLASSES), pretrained=True).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Check if resuming from checkpoint
    start_epoch = 1
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if RESUME_TRAINING:
        checkpoint_path = find_latest_checkpoint(MODELS_DIR)
        if checkpoint_path:
            print("\n" + "=" * 80)
            print("RESUMING TRAINING FROM CHECKPOINT")
            print("=" * 80)
            loaded_epoch, history, best_val_acc = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
            start_epoch = loaded_epoch + 1
            # Use the same timestamp from the checkpoint filename
            timestamp = checkpoint_path.stem.split('_')[-2] + '_' + checkpoint_path.stem.split('_')[-1]
            print(f"✓ Will continue training from epoch {start_epoch}")
            print("=" * 80)
            print()
        else:
            print("✓ No checkpoint found. Starting fresh training.")
            print()
    
    print("=" * 80)
    if start_epoch > 1:
        print(f"Resuming Training from Epoch {start_epoch}/{NUM_EPOCHS}")
    else:
        print("Starting Training")
    print("=" * 80)
    print("Press Ctrl+C to stop training early")
    print()
    
    # Check if already completed
    if start_epoch > NUM_EPOCHS:
        print(f"✓ Training already completed ({start_epoch-1}/{NUM_EPOCHS} epochs)")
        print(f"✓ Best accuracy: {best_val_acc:.2f}%")
        print("\nTo train more epochs, increase NUM_EPOCHS in the script.")
        return
    
    # Training loop
    try:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = validate_epoch(
                model, val_loader, criterion, device
            )
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = MODELS_DIR / f"asl_model_best_{timestamp}.pth"
                save_checkpoint(model, optimizer, epoch, history, best_val_acc, best_model_path)
                print(f"  ★ New best model! Validation accuracy: {best_val_acc:.2f}%")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = MODELS_DIR / f"asl_model_epoch_{epoch}_{timestamp}.pth"
                save_checkpoint(model, optimizer, epoch, history, best_val_acc, checkpoint_path)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Training interrupted by user!")
        print("=" * 80)
        print(f"Stopped at epoch {epoch}/{NUM_EPOCHS}")
        print(f"Best validation accuracy so far: {best_val_acc:.2f}%")
        
        # Save final checkpoint
        if len(history['train_loss']) > 0:
            interrupted_path = MODELS_DIR / f"asl_model_interrupted_epoch{epoch}_{timestamp}.pth"
            save_checkpoint(model, optimizer, epoch, history, best_val_acc, interrupted_path)
            print(f"\n✓ Saved interrupted model to: {interrupted_path}")
        print("\nYou can resume training later by loading this checkpoint.")
        print("=" * 80)
        return  # Exit gracefully
    
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print()
    
    # Final evaluation
    print("Running final evaluation...")
    model.eval()
    final_loss, final_acc, final_preds, final_labels = validate_epoch(
        model, val_loader, criterion, device
    )
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(
        final_labels, final_preds,
        target_names=CLASSES,
        digits=3
    )
    print(report)
    
    # Save classification report
    report_path = RESULTS_DIR / f"classification_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Classification report saved to {report_path}")
    
    # Plot training history
    history_plot_path = RESULTS_DIR / f"training_history_{timestamp}.png"
    plot_training_history(history, history_plot_path)
    
    # Plot confusion matrix
    cm_plot_path = RESULTS_DIR / f"confusion_matrix_{timestamp}.png"
    plot_confusion_matrix(final_labels, final_preds, cm_plot_path)
    
    # Save training config and results
    config = {
        'timestamp': timestamp,
        'device': str(device),
        'num_classes': len(CLASSES),
        'classes': CLASSES,
        'total_images': len(full_dataset),
        'train_images': len(train_dataset),
        'val_images': len(val_dataset),
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'best_val_acc': float(best_val_acc),
        'final_val_acc': float(final_acc),
        'final_val_loss': float(final_loss),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'class_distribution': distribution
    }
    
    config_path = RESULTS_DIR / f"training_config_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Training config saved to {config_path}")
    
    print("\n" + "=" * 80)
    print("All results saved!")
    print(f"  - Best model: {best_model_path}")
    print(f"  - Training history: {history_plot_path}")
    print(f"  - Confusion matrix: {cm_plot_path}")
    print(f"  - Classification report: {report_path}")
    print(f"  - Config: {config_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

