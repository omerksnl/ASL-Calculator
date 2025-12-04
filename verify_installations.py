"""
Verification Script for FLASLR Project
Tests all required software installations and dependencies.
Run this script to ensure your environment is ready for development.
"""

import sys
import importlib

def check_python_version():
    """Check if Python version is 3.11 or compatible."""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    # Check if we're in the venv311 environment
    venv_path = getattr(sys, 'prefix', '')
    in_venv311 = 'venv311' in venv_path
    
    if version.major == 3 and version.minor >= 11:
        if version.minor == 13:
            print("⚠ Python 3.13 detected - MediaPipe requires Python 3.11")
            if not in_venv311:
                print("  → Please activate venv311: venv311\\Scripts\\activate")
                print("  → Then run this script again\n")
            else:
                print("  → Using venv311 environment\n")
        else:
            print("✓ Python version is compatible (3.11+)\n")
        return True
    else:
        print("✗ Python version should be 3.11 or higher\n")
        return False

def check_package(package_name, import_name=None, version_attr=None):
    """Check if a package is installed and optionally print version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = "Unknown"
        
        if version_attr:
            version = getattr(module, version_attr, "Unknown")
        elif hasattr(module, '__version__'):
            version = module.__version__
        
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        return False

def check_opencv():
    """Check OpenCV installation and webcam access."""
    try:
        import cv2
        print(f"✓ OpenCV (cv2): {cv2.__version__}")
        
        # Try to access webcam
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Webcam access: Available")
            cap.release()
        else:
            print("⚠ Webcam access: No camera detected (may be in use)")
        return True
    except ImportError:
        print("✗ OpenCV (cv2): NOT INSTALLED")
        return False

def check_mediapipe():
    """Check MediaPipe installation."""
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe: {mp.__version__}")
        
        # Try to initialize hands module
        mp_hands = mp.solutions.hands
        print("✓ MediaPipe Hands: Available")
        return True
    except ImportError:
        print("✗ MediaPipe: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"⚠ MediaPipe: Installed but error initializing - {e}")
        return False

def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    try:
        import torch
        import torchvision
        
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ Torchvision: {torchvision.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA: Available (Device: {torch.cuda.get_device_name(0)})")
        else:
            print("ℹ CUDA: Not available (CPU mode - fine for this project)")
        
        # Check MobileNetV2 availability (without loading weights to avoid download delay)
        try:
            from torchvision import models
            # Just check if the function exists, don't load weights
            if hasattr(models, 'mobilenet_v2'):
                print("✓ MobileNetV2: Available via torchvision (model class found)")
            else:
                print("⚠ MobileNetV2: Model class not found in torchvision")
        except Exception as e:
            print(f"⚠ MobileNetV2: Error checking - {e}")
        
        return True
    except ImportError:
        print("✗ PyTorch: NOT INSTALLED")
        return False

def check_flower():
    """Check Flower framework installation."""
    try:
        import flwr
        print(f"✓ Flower (flwr): {flwr.__version__}")
        return True
    except ImportError:
        print("✗ Flower (flwr): NOT INSTALLED")
        return False
    except AttributeError:
        # Some versions don't have __version__
        print("✓ Flower (flwr): Installed (version info unavailable)")
        return True

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("FLASLR Project - Installation Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Python version
    print("1. Python Environment:")
    print("-" * 60)
    results.append(check_python_version())
    
    # Core ML and FL frameworks
    print("2. Machine Learning & Federated Learning:")
    print("-" * 60)
    results.append(check_pytorch())
    results.append(check_flower())
    print()
    
    # Computer Vision
    print("3. Computer Vision Libraries:")
    print("-" * 60)
    results.append(check_opencv())
    results.append(check_mediapipe())
    print()
    
    # Supporting libraries
    print("4. Supporting Libraries:")
    print("-" * 60)
    results.append(check_package("NumPy", "numpy"))
    results.append(check_package("Pillow", "PIL"))
    results.append(check_package("Matplotlib", "matplotlib"))
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All checks passed ({passed}/{total})")
        print("\nYour environment is ready for FLASLR development!")
        return 0
    else:
        print(f"⚠ Some checks failed ({passed}/{total} passed)")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

