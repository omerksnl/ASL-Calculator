"""
Script to prepare IID and Non-IID data distributions for federated learning
Files are COPIED (originals remain in master_data)
"""
import shutil
from pathlib import Path
import random

# Configuration
SOURCE_DIR = Path("data/master_dataset/master_data")
IID_DIR = Path("data/master_dataset/master_iid")
NONIID_DIR = Path("data/master_dataset/master_noniid")

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'divide', 'equals', 'minus', 'multiply', 'plus']

random.seed(42)  # For reproducibility


def prepare_iid_data():
    """
    IID: Each client (both on Pi) gets ~50% of each class (similar distribution)
    Client 1: Balanced distribution
    Client 2: Balanced distribution
    """
    print("\n" + "=" * 60)
    print("Preparing IID (balanced) Data Distribution")
    print("Both clients on Raspberry Pi")
    print("=" * 60)
    
    client1_dir = IID_DIR / "client1"
    client2_dir = IID_DIR / "client2"
    
    # Create directories
    for cls in CLASSES:
        (client1_dir / cls).mkdir(parents=True, exist_ok=True)
        (client2_dir / cls).mkdir(parents=True, exist_ok=True)
    
    total_client1 = 0
    total_client2 = 0
    
    for cls in CLASSES:
        class_dir = SOURCE_DIR / cls
        if not class_dir.exists():
            print(f"⚠ Skipping {cls} - no data found")
            continue
        
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)
        
        # Split 50/50
        split_point = len(images) // 2
        
        client1_images = images[:split_point]
        client2_images = images[split_point:]
        
        # Copy to client directories
        for img in client1_images:
            shutil.copy2(img, client1_dir / cls / img.name)
        
        for img in client2_images:
            shutil.copy2(img, client2_dir / cls / img.name)
        
        total_client1 += len(client1_images)
        total_client2 += len(client2_images)
        
        print(f"  {cls:8s}: Client1={len(client1_images):3d}, Client2={len(client2_images):3d}")
    
    print(f"\n✓ IID Data Ready:")
    print(f"  Client 1 total: {total_client1} images")
    print(f"  Client 2 total: {total_client2} images")
    print(f"  Distribution: Balanced (each client has similar class distribution)")


def prepare_noniid_data():
    """
    Non-IID: Completely different data types
    - Client 1 (Pi): ALL digits (0-9)
    - Client 2 (Pi): ALL operators (divide, equals, minus, multiply, plus)
    Both clients run on Raspberry Pi
    """
    print("\n" + "=" * 60)
    print("Preparing Non-IID (completely skewed) Data Distribution")
    print("Client 1: Digits (0-9)")
    print("Client 2: Operators (+, -, ×, ÷, =)")
    print("Both clients on Raspberry Pi")
    print("=" * 60)
    
    client1_dir = NONIID_DIR / "client1"
    client2_dir = NONIID_DIR / "client2"
    
    # Create directories
    for cls in CLASSES:
        (client1_dir / cls).mkdir(parents=True, exist_ok=True)
        (client2_dir / cls).mkdir(parents=True, exist_ok=True)
    
    # Define which client gets which classes
    # Client 1: Only digits (0-9)
    # Client 2: Only operators
    client1_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    client2_classes = ['divide', 'equals', 'minus', 'multiply', 'plus']
    
    total_client1 = 0
    total_client2 = 0
    
    # Process Client 1 classes (digits)
    print("\nClient 1 (Digits):")
    for cls in client1_classes:
        class_dir = SOURCE_DIR / cls
        if not class_dir.exists():
            print(f"  ⚠ Skipping {cls} - no data found")
            continue
        
        images = list(class_dir.glob("*.jpg"))
        
        # Copy ALL digit images to client1
        for img in images:
            shutil.copy2(img, client1_dir / cls / img.name)
        
        total_client1 += len(images)
        print(f"  {cls:8s}: {len(images):3d} images")
    
    # Process Client 2 classes (operators)
    print("\nClient 2 (Operators):")
    for cls in client2_classes:
        class_dir = SOURCE_DIR / cls
        if not class_dir.exists():
            print(f"  ⚠ Skipping {cls} - no data found")
            continue
        
        images = list(class_dir.glob("*.jpg"))
        
        # Copy ALL operator images to client2
        for img in images:
            shutil.copy2(img, client2_dir / cls / img.name)
        
        total_client2 += len(images)
        print(f"  {cls:8s}: {len(images):3d} images")
    
    print(f"\n✓ Non-IID Data Ready:")
    print(f"  Client 1 total: {total_client1} images (digits 0-9)")
    print(f"  Client 2 total: {total_client2} images (operators)")
    print(f"  Distribution: COMPLETELY SKEWED - Different data types per client")


def main():
    print("=" * 60)
    print("IID/Non-IID Data Preparation for Federated Learning")
    print("Files will be COPIED (originals kept in master_data)")
    print("=" * 60)
    print()
    
    # Check source directory
    if not SOURCE_DIR.exists():
        print(f"✗ Error: Source directory not found: {SOURCE_DIR}")
        print("  Please ensure you have data in data/master_dataset/master_data/")
        return
    
    # Count total images
    total_images = sum(len(list((SOURCE_DIR / cls).glob("*.jpg"))) 
                      for cls in CLASSES if (SOURCE_DIR / cls).exists())
    print(f"\n✓ Found {total_images} total images in {SOURCE_DIR}")
    
    # Prepare both distributions
    prepare_iid_data()
    prepare_noniid_data()
    
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\n✓ Files have been copied!")
    print(f"✓ Original master_data folder is intact")
    
    print("\nNext steps:")
    print("1. Copy these folders to your Raspberry Pi:")
    print(f"   - {IID_DIR}/")
    print(f"   - {NONIID_DIR}/")
    print("\n2. Run federated learning experiments:")
    print("\n   IID Experiment (balanced - each client has 50% of all classes):")
    print("     Pi Terminal 1: python src/client.py --server PC_IP:8080 --data-dir data/master_dataset/master_iid/client1")
    print("     Pi Terminal 2: python src/client.py --server PC_IP:8080 --data-dir data/master_dataset/master_iid/client2")
    print("\n   Non-IID Experiment (completely skewed - digits vs operators):")
    print("     Pi Terminal 1: python src/client.py --server PC_IP:8080 --data-dir data/master_dataset/master_noniid/client1  (digits)")
    print("     Pi Terminal 2: python src/client.py --server PC_IP:8080 --data-dir data/master_dataset/master_noniid/client2  (operators)")
    print("=" * 60)


if __name__ == "__main__":
    main()

