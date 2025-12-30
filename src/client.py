import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from pathlib import Path
import cv2
import numpy as np
from collections import OrderedDict
import argparse

# Configuration
DATA_DIR = Path("data/master_dataset/master_data")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = 224
NUM_WORKERS = 2  # Reduced for Pi compatibility
LOCAL_EPOCHS = 5  # Epochs per round

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
        
        print(f"✓ Client loaded {len(self.samples)} images from {len(CLASSES)} classes")
        
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


class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    def get_parameters(self, config):
        """Return model parameters as a list of NumPy ndarrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy ndarrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model on the local dataset"""
        print(f"\n{'='*60}")
        print(f"Starting local training for {LOCAL_EPOCHS} epochs")
        print(f"{'='*60}")
        
        self.set_parameters(parameters)
        self.model.train()
        
        for epoch in range(LOCAL_EPOCHS):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"  Epoch [{epoch+1}/{LOCAL_EPOCHS}] "
                          f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                          f"Loss: {running_loss/(batch_idx+1):.4f} "
                          f"Acc: {100*correct/total:.2f}%")
            
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100 * correct / total
            print(f"✓ Epoch {epoch+1} completed: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation dataset"""
        print(f"\n{'='*60}")
        print("Evaluating model on local validation set")
        print(f"{'='*60}")
        
        self.set_parameters(parameters)
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100 * correct / total
        
        print(f"✓ Validation: Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return float(val_loss), len(self.val_loader.dataset), {"accuracy": float(val_acc)}


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client for ASL Recognition')
    parser.add_argument('--server', type=str, default='localhost:8080',
                       help='Server address (e.g., localhost:8080 or 192.168.1.100:8080)')
    parser.add_argument('--data-dir', type=str, default='data/master_dataset/master_data',
                       help='Path to training data directory')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Federated Learning Client - ASL Gesture Recognition")
    print("=" * 80)
    print(f"Server address: {args.server}")
    print(f"Data directory: {args.data_dir}")
    print()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load dataset
    print("Loading local dataset...")
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"✗ Error: Data directory not found: {data_dir}")
        print("  Please ensure data is available on this client.")
        return
    
    full_dataset = ASLDataset(data_dir, transform=None)
    
    # Split dataset (80/20 train/val)
    val_size = int(len(full_dataset) * 0.2)
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
    print(f"✓ Model ready with {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Create Flower client
    print("=" * 80)
    print("Connecting to Federated Learning server...")
    print(f"Server: {args.server}")
    print("=" * 80)
    print()
    
    client = FlowerClient(model, train_loader, val_loader, device)
    
    # Start Flower client
    try:
        fl.client.start_numpy_client(
            server_address=args.server,
            client=client
        )
    except KeyboardInterrupt:
        print("\n\nClient interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Is the server running?")
        print("  2. Is the server address correct?")
        print("  3. Check firewall settings")
    
    print("\n" + "=" * 80)
    print("Client disconnected")
    print("=" * 80)


if __name__ == "__main__":
    main()

