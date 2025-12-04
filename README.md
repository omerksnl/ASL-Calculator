# Federated Sign Language Calculator (FLASLR)

A federated learning project that trains a sign language recognition model across edge devices (Raspberry Pi 5) to recognize 15 ASL gestures (digits 0-9 and operators +, -, *, /, =) for real-time calculator operations.

## Project Overview

This project implements a federated learning system to train a MobileNetV2-based ASL gesture recognition model. The system compares performance between IID (Independent and Identically Distributed) and Non-IID data distributions across federated clients.

**Research Question:** How does non-IID data distribution affect model performance compared to IID data in federated learning scenarios?

## Features

- **Federated Learning:** Train models across multiple Raspberry Pi devices using Flower framework
- **ASL Recognition:** Recognize 15 static gestures (0-9 digits + 5 operators)
- **Real-time Calculator:** Live webcam-based calculator using trained model
- **Non-IID Analysis:** Compare FedAvg vs FedProx strategies on non-IID data

## Project Structure

```
FLASLR/
├── definition.md              # Complete project definition and plan
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── verify_installations.py     # Script to test all installations
├── src/
│   ├── create_data.py          # Week 2: Data collection tool
│   ├── train_local.py          # Week 3: Baseline local training
│   ├── client.py               # Week 4: Federated learning client
│   ├── server.py               # Week 4: Federated learning server
│   └── live_demo.py            # Week 7: Real-time calculator demo
├── data/
│   ├── master_dataset/         # Master dataset for training
│   └── test_set/               # Test dataset for evaluation
└── models/                     # Trained model checkpoints
```

## Installation

### Prerequisites

- **Python 3.11** (required - MediaPipe doesn't support Python 3.13+ yet)
- Webcam for data collection
- (Optional) 2x Raspberry Pi 5 devices for federated training

### Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd FLASLR
   ```

2. **Create Python 3.11 virtual environment:**
   ```bash
   # On Windows (using Python Launcher):
   py -3.11 -m venv venv311
   
   # Or if you have python3.11 directly:
   python3.11 -m venv venv311
   ```

3. **Activate virtual environment:**
   ```bash
   # On Windows:
   venv311\Scripts\activate
   
   # On Linux/Mac:
   source venv311/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installations:**
   ```bash
   # Make sure you're using the venv311 Python:
   python verify_installations.py
   
   # Or explicitly:
   venv311\Scripts\python.exe verify_installations.py
   ```

## Usage

### Week 2: Data Collection

Collect ASL gesture images using the data collection tool:

```bash
python src/create_data.py
```

**Controls:**
- Press `0-9` to save digit gestures
- Press `+, -, *, /, =` to save operator gestures
- Press `c` to view current image counts
- Press `q` to quit

Images are saved to `data/master_dataset/master_data/` organized by class folders.

### Week 3: Local Training (Baseline)

Train a baseline model locally:

```bash
python src/train_local.py
```

### Week 4-6: Federated Training

**Server (on laptop):**
```bash
python src/server.py
```

**Client (on each Raspberry Pi):**
```bash
python src/client.py --data-dir <path-to-client-data>
```

### Week 7: Real-time Calculator Demo

Run the live calculator application:

```bash
python src/live_demo.py
```

## Development Timeline

- **Week 1-3:** Setup, data collection, and baseline training
- **Week 4-6:** Federated training (IID vs Non-IID comparison)
- **Week 7-8:** Real-time application and finalization

See `definition.md` for complete project details and timeline.

## Technologies

- **PyTorch:** Deep learning framework
- **Flower (flwr):** Federated learning framework
- **MediaPipe:** Hand detection and tracking
- **OpenCV:** Computer vision and webcam capture
- **MobileNetV2:** Lightweight CNN architecture

## Authors

- Ömer Kaan Şanal - S040071
- Metin Bora Baysal - S033620
- Ozan Yelaldı - S040017

## License

This project is for academic/research purposes.

