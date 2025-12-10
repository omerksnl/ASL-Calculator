# Federated Sign Language Calculator - Project Definition

**Submission Date:** November 9, 2025

**Group Members:**
- Ömer Kaan Şanal - S040071
- Metin Bora Baysal - S033620
- Ozan Yelaldı - S040017

---

## 1. Project Topic & Objectives

### 1.1. Project Overview

This project will create a Federated Sign Language Calculator that recognizes 15 static American Sign Language (ASL) gestures: the digits 0-9 and five mathematical operators (+, -, *, /, =) for basic algebraic operations.

We will train a global model using two Raspberry Pi 5 (4 GB VRAM) devices as federated clients, each holding its own private dataset. A key research question we aim to answer is: **How does non-IID data distribution affect model performance compared to IID data in federated learning scenarios?**

The final product will be a real-time application running on a laptop that uses the federated model to perform simple calculations (e.g., recognizing "5", "+", "3", "=") based on live video feed from a webcam, demonstrating practical deployment of federated learning in edge computing environments.

### 1.2. Objectives

1. **Federated Learning Deployment:** Successfully deploy and manage a complete FL training process using the Flower framework, with a laptop serving as the aggregation server and two Raspberry Pi devices as federated clients. This will demonstrate practical federated learning implementation in a resource-constrained edge computing environment.

2. **Non-IID vs. IID Performance Analysis:** Train a robust model on a challenging Non-IID dataset distribution (e.g., Pi 1 contains only numeric gestures 0-9, Pi 2 contains only operator gestures +, -, *, /, =) and compare its performance to a baseline IID model where each client receives a balanced, random distribution of all 15 classes. This addresses a fundamental challenge in federated learning research.

3. **Real-Time Application Development:** Create a production-ready, real-time "calculator" demo application that visually recognizes and processes ASL gestures from live webcam feed, demonstrating the practical viability of federated learning models in real-world deployment scenarios.

---

## 2. Background Research

### 2.1. Federated Learning Fundamentals

Federated Learning (FL) is a distributed machine learning paradigm that enables model training across multiple decentralized edge devices while keeping data local [1]. This approach addresses privacy concerns in centralized machine learning. In FL, only model updates are shared, not the raw data itself, providing a fundamental mechanism that preserves privacy.

The standard federated learning algorithm, Federated Averaging (FedAvg), aggregates local model updates from participating clients using weighted averaging based on the number of training samples per client [1]. However, FedAvg assumes an IID data distribution across clients, which is rarely the case in real-world scenarios. When data is non-IID—meaning different clients have different data distributions—FedAvg can suffer from convergence issues and reduced model accuracy.

### 2.2. Non-IID Data Challenges in Federated Learning

The Non-IID data problem in federated learning is a well-known challenge. Non-IID data can significantly degrade model performance compared to centralized training. This happens because:

- **Client Drift:** Local models trained on different data distributions diverge significantly from the global model
- **Weight Divergence:** Averaging divergent local models can result in a global model that performs poorly on all clients
- **Convergence Issues:** The global model may fail to converge or converge to sub-optimal solutions

To address these challenges, several algorithms have been proposed:

- **FedProx:** Adds a proximal term to the local objective function to prevent excessive client drift [3]
- **SCAFFOLD:** Uses control variates to correct for client drift
- **FedNova:** Normalizes local updates to account for different numbers of local epochs

We may use FedProx rather than FedAvg as it is readily available in the Flower framework and has shown effectiveness in handling non-IID data distributions [3].

### 2.3. Sign Language Recognition

Sign language recognition has been an active area of computer vision research. Modern deep learning approaches have significantly improved recognition accuracy:

- Convolutional Neural Networks (CNNs) have shown excellent performance in static gesture recognition
- Transfer Learning using pre-trained models like MobileNet, ResNet, and EfficientNet has proven effective for sign language recognition tasks
- MediaPipe Hands provides real-time hand tracking and landmark detection, enabling robust hand segmentation for gesture recognition [4]

### 2.4. Edge Computing and Resource Constraints

Edge computing brings machine learning inference and training closer to data sources, reducing latency and bandwidth requirements. However, edge devices like Raspberry Pi have limited computational resources (CPU, memory, storage), necessitating:

- **Model Compression:** Using lightweight architectures like MobileNetV2 [5]
- **Quantization:** Reducing model precision to decrease memory footprint
- **Efficient Frameworks:** Using optimized libraries like PyTorch Mobile and TensorFlow Lite

### 2.5. Related Work

Several studies have explored federated learning for computer vision tasks. However, few studies have specifically addressed federated learning for sign language recognition with non-IID data distributions, making our project a novel contribution to the field.

---

## 3. Software & Hardware Selection

### 3.1. Hardware Selection

#### 3.1.1. Client Devices: Raspberry Pi 5

**Device:** 1x Raspberry Pi 5 (4GB RAM)

**Specifications:**
- CPU: Broadcom BCM2712 quad-core Cortex-A76 @ 2.4GHz
- RAM: 4GB LPDDR4X-4267
- GPU: VideoCore VII
- Storage: MicroSD card (minimum 32GB recommended)
- Connectivity: Gigabit Ethernet, Wi-Fi 5, Bluetooth 5.0
- Cost: ~3,240.10 TL per device (Robotistan)
- Additional costs: Power supply (~716.26 TL in Robotistan), MicroSD card (~180 TL in Amazon)

**Justification:** The Raspberry Pi 5 represents an ideal edge computing platform for our project. It provides sufficient computational power for local model training while maintaining realistic resource constraints that mirror real-world edge devices (smartphones, IoT devices, embedded systems). The 4GB RAM limitation forces us to optimize our model architecture and training procedures, making our solution more practical and deployable. The CNN we'll train will already be able to track hand movement. 

**Training Approach:** We will use a single Raspberry Pi 5 and run federated learning clients sequentially (not simultaneously). This approach:
- Reduces hardware costs from ~8,272 TL to ~4,136 TL (50% cost savings)
- Still demonstrates the federated learning concept effectively
- Allows us to simulate two different clients by running client.py with different datasets sequentially
- Maintains the same experimental validity for comparing IID vs Non-IID data distributions

#### 3.1.2. Server Device: Laptop

**Device:** Standard laptop (Windows + 16GB RAM)

**Role:** Central aggregation server for federated learning

**Requirements:**
- Python 3.11 environment
- Network connectivity for client-server communication
- Has enough storage for datasets and model

### 3.2. Core Model Architecture Selection

#### 3.2.1. MobileNetV2 Architecture

**Model:** MobileNetV2 (via PyTorch/torchvision)

**Architecture Details:**
- Depthwise separable convolutions for efficiency
- Inverted residual blocks with linear bottlenecks
- Pre-trained on ImageNet dataset (1.4M images, 1000 classes)
- Model size: ~14MB (FP32), suitable for edge deployment
- Input size: 224×224×3 (RGB images)

**Justification:** MobileNetV2 was specifically designed by Google for mobile and edge devices, achieving state-of-the-art performance with minimal computational overhead [5]. The architecture uses depthwise separable convolutions, which reduce the number of parameters and operations by approximately 8-9× compared to standard convolutions while maintaining accuracy.

We employ transfer learning by using MobileNetV2 pre-trained on ImageNet, which provides:

- **Feature Extraction:** The model has learned rich visual features (edges, textures, shapes) from millions of natural images
- **High Accuracy:** Transfer learning typically achieves 90%+ accuracy with small datasets (200-300 images per class)
- **Resource Efficiency:** Pre-trained models converge faster, reducing training time and energy consumption on edge devices

**Alternative Considered:** We initially considered training a custom lightweight CNN from scratch. However, this approach was dismissed because:
- Training from scratch requires significantly more data (thousands of images per class)
- Longer training time (days vs. hours)
- Lower accuracy
- Higher computational cost during training

### 3.3. Software Stack & Installation Status

| Category | Software / Tool | Version | Purpose | Status |
|----------|----------------|---------|---------|--------|
| OS (Clients) | Raspberry Pi OS (64-bit) | Latest | Operating system for Raspberry Pi clients. Provides optimized Python environment and hardware drivers. | Not Installed Yet |
| Programming Language | Python | 3.11.9 | Core programming language. Version 3.11 chosen for MediaPipe compatibility and performance improvements. | Installed |
| FL Framework | Flower (flwr) | Latest | Lightweight, framework-agnostic federated learning library. Manages server-client communication, model aggregation (FedAvg, FedProx), and distributed training coordination. | Installed |
| ML Framework | PyTorch | Latest | Deep learning framework for building, training, and inference. Provides MobileNetV2 via torchvision models. | Installed |
| Computer Vision | OpenCV (cv2) | Latest | Computer vision library for webcam capture, image preprocessing, real-time video processing, and UI rendering (text overlays, bounding boxes). | Installed |
| Hand Tracking | MediaPipe Hands | Latest | Google's pre-trained hand detection and landmark detection model. Detects hand bounding boxes in real-time, enabling accurate hand segmentation before gesture classification. Dramatically improves accuracy by focusing on the hand region only. | Installed |
| Image Processing | Pillow (PIL) | Latest | Python Imaging Library for image loading, resizing, and format conversion. Required by PyTorch's image data loaders. | Installed |
| Numerical Computing | NumPy | Latest | Fundamental library for numerical operations. Required by Flower for model weight manipulation and array operations. | Installed |
| Version Control | Git / GitHub | Latest | Source code version control and collaboration platform. Repository created for project codebase management. | Repository Created |
| Development Environment | Virtual Environment (venv) | Python 3.13 | Isolated Python environment for dependency management. Prevents package conflicts. | Created |

**Software Installed:**
- GitHub Repo Made
- Flower Test
- MediaPipe Test
- OpenCV Test (cv2.cvtColor function that makes webcam footage black+white - proof that it works)

---

## 4. Project Plan (8-Week Timeline)

### Phase 1: Setup, Data Pipeline, & Prototyping (Weeks 1-3)

#### Week 1
- **Task:** Finalize/test all software installations on all 3 devices (Laptop, 2x Pis)
- **Task:** Test network connectivity between all devices

#### Week 2
- **Task:** Develop the `create_data.py` script. This script will use OpenCV to open the webcam and MediaPipe to detect the hand's bounding box in real-time
- **Result:** A script that, upon a key press, saves a cropped image of the hand to the correct class folder

#### Week 3
- **Task 1:** Use `create_data.py` to build the "Master Dataset" on the laptop
  - We will collect ~200-300 cropped images for each of the 15 gestures
  - Then, create a separate test set (~50 images/class) that will only stay on the laptop for final evaluation
- **Task 2:** Write `train_local.py`. This is a non-federated script to train the MobileNetV2 model on the Master Dataset
- **Result:** A baseline trained model and its accuracy score on the test set. This will help us see the initial training accuracy

### Phase 2: Federated Training & Model Refinement (Weeks 4-6)

#### Week 4
- **Task:** Create the Non-IID datasets
  - `pi1_data_noniid` (Classes 0-9)
  - `pi2_data_noniid` (Classes +, -, *, /, =)
- **Task:** Transfer these datasets to the Pi (will be used sequentially as Client 1 and Client 2)
- **Task:** Refactor `train_local.py` into `client.py` and `server.py` using the Flower NumPyClient template
- **Result:** A functional `client.py` and `server.py`

#### Week 5
- **Task:** Run the first full federated training using the standard FedAvg algorithm
  - `server.py` on Laptop
  - `client.py` with `pi1_data_noniid` on Pi (run as Client 1)
  - After Client 1 completes, run `client.py` with `pi2_data_noniid` on Pi (run as Client 2)
  - Repeat this sequential process for each FL round
- **Note:** The Pi will alternate between being Client 1 and Client 2 in each round. Clients run sequentially, not simultaneously.
- **Task:** Let the system run for 10-20 FL rounds
- **Result:** The first global model: `global_model_noniid.pth`

#### Week 6
- **Task:** Create and distribute "fair" IID datasets (split into two parts: each part gets a random 50% of all 15 classes for sequential client training)
- **Task:** Re-run the FL training
- **First Result:** The second global model: `global_model_iid.pth`
- **Second Result:** An accuracy graph comparing `local_baseline`, `global_model_iid`, and `global_model_noniid` on the laptop's test set. This is the core result of our experiment

**In Addition to Week 5-6:** If the FedAvg model is incompetent or has low accuracy due to the non-IID data, we plan to solve this by changing the server's strategy to FedProx. If necessary, we will re-run the Week 5 experiment using FedProx to create an improved `global_model_noniid_fedprox.pth`. We think that the aggregation type won't matter in IID data.

**FedAvg Logic with non-IID Data:**
- You give Pi 1 only numbers (0-9). It becomes a "Number Specialist."
- You give Pi 2 only symbols (+, -). It becomes a "Symbol Specialist."
- FedAvg only averages these two specialist brains. The result is a "confused" model that is bad at both tasks, giving you low accuracy.

**FedProx Logic with non-IID Data:**
- You give Pi 1 the numbers, but FedProx adds a "leash," telling it, "Don't drift too far from the global model."
- You give Pi 2 the symbols, but FedProx adds the same "leash."
- Because the leash keeps both clients close to the original, they don't become extreme specialists. When the server averages them, they merge cleanly into a single, high-accuracy model.

### Phase 3: Final Application (Weeks 7-8)

#### Week 7
- **Task:** Develop the `live_demo.py` script
  - **Logic:**
    - Load the best global model (will probably be `global_model_noniid.pth`)
    - Start OpenCV/MediaPipe to find and crop the hand
    - Feed the cropped image to the model for a prediction
    - Implement the simple calculator state machine
- **Result:** A working, real-time ASL calculator application

#### Week 8
- **Task:** Record a video of the final demo. Clean code and add comments
- **Result:** Prepare the final project report and presentation slides, highlighting the privacy benefits and the IID vs. Non-IID results

---

## 5. References

[1] Flower Framework Documentation. (2024). Flower: A Friendly Federated Learning Framework. https://flower.dev/

[2] Flower Framework - Strategies. (2024). Federated Averaging (FedAvg) Strategy. https://flower.dev/docs/framework/how-to-use-strategies.html

[3] Flower Framework - FedProx. (2024). FedProx Strategy Implementation. https://flower.dev/docs/apiref/flwr/server/strategy/fedprox.html

[4] Google MediaPipe. (2024). MediaPipe Hands: On-device Real-time Hand Tracking. https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

[5] PyTorch Documentation. (2024). MobileNetV2 Model Architecture. https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html

[6] Raspberry Pi Foundation. (2024). Raspberry Pi 5 Documentation. https://www.raspberrypi.com/documentation/

[7] OpenCV Documentation. (2024). OpenCV Python Tutorials. https://docs.opv.org/4.x/d6/d00/tutorial_py_root.html

[8] PyTorch Documentation. (2024). PyTorch Transfer Learning Tutorial. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

[9] Custom Hand Gesture Recognition with Hand Landmarks Using Google's Mediapipe + OpenCV in Python (2022). Ivan Goncharov. http://www.youtube.com/watch?v=a99p_fAr6e4