# FLASLR Setup Guide

## Important: Python Version Requirement

**MediaPipe requires Python 3.11** - it does not support Python 3.13+ yet.

If you have Python 3.13 installed (like the default on your system), you need to use Python 3.11 in a virtual environment.

## Quick Setup Steps

### 1. Check if Python 3.11 is available

```bash
# On Windows:
py -3.11 --version

# Should show: Python 3.11.x
```

If Python 3.11 is not installed, download it from [python.org](https://www.python.org/downloads/) and install it.

### 2. Create Python 3.11 Virtual Environment

```bash
# On Windows:
py -3.11 -m venv venv311

# On Linux/Mac:
python3.11 -m venv venv311
```

### 3. Activate the Virtual Environment

```bash
# On Windows (PowerShell):
venv311\Scripts\Activate.ps1

# On Windows (CMD):
venv311\Scripts\activate.bat

# On Linux/Mac:
source venv311/bin/activate
```

You should see `(venv311)` in your terminal prompt.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch & Torchvision
- Flower (federated learning framework)
- OpenCV
- MediaPipe (requires Python 3.11)
- NumPy, Pillow, Matplotlib

### 5. Verify Installation

```bash
python verify_installations.py
```

All checks should pass (8/8).

## Using the Project

**Always activate the venv311 environment before running scripts:**

```bash
# Activate first:
venv311\Scripts\activate  # Windows
# or
source venv311/bin/activate  # Linux/Mac

# Then run scripts:
python src/create_data.py
python src/train_local.py
# etc.
```

## Troubleshooting

### MediaPipe not found
- Make sure you're using Python 3.11 in venv311
- Check: `python --version` should show 3.11.x
- If it shows 3.13, activate venv311: `venv311\Scripts\activate`

### Packages not found after activation
- Reinstall: `pip install -r requirements.txt`
- Make sure venv311 is activated (check prompt shows `(venv311)`)

### Webcam not working
- Check if another application is using the webcam
- Try closing other camera apps
- Verify webcam works in other applications

