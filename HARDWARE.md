# Hardware Requirements

## Required Hardware

### 1. Raspberry Pi 5 (1x)
- **Model:** Raspberry Pi 5 (4GB RAM)
- **Cost:** ~3,240.10 TL (Robotistan)
- **Purpose:** Federated learning client device

### 2. Raspberry Pi 5 Accessories
- **Power Supply:** ~716.26 TL (Robotistan)
- **MicroSD Card:** 
  - **16GB:** Minimum, but tight (may work if careful)
  - **32GB:** Recommended (safer, more comfortable)
  - Cost: ~100-180 TL depending on size
- **Total Accessories:** ~816-896 TL

### 3. Server Device (Already Available)
- **Device:** Laptop (Windows + 16GB RAM)
- **Purpose:** Central aggregation server for federated learning
- **Status:** ✅ Already available

## Total Cost

**Raspberry Pi 5 Setup:**
- Raspberry Pi 5 (4GB): ~3,240.10 TL
- Power Supply: ~716.26 TL
- MicroSD Card: ~100-180 TL (16GB or 32GB)
- **Total: ~4,056-4,136 TL**

## Storage Requirements

**Breakdown of storage needs on Raspberry Pi:**

1. **Raspberry Pi OS (64-bit):** ~8-10GB
2. **Python 3.11 + Packages:**
   - PyTorch: ~1.5-2GB
   - Torchvision: ~200MB
   - Flower, MediaPipe, OpenCV, etc.: ~300MB
   - Total packages: ~3GB
3. **Datasets (on Pi):**
   - pi1_data_noniid (0-9): ~300-450MB
   - pi2_data_noniid (operators): ~150-225MB
   - Total: ~500-700MB
4. **Model files:** ~50-100MB
5. **System files, logs, temp space:** ~2-3GB

**Total: ~13-17GB**

### Recommendation

- **16GB:** ⚠️ **Tight but possible** - You'll need to:
  - Clean up unused packages
  - Remove old logs/temp files regularly
  - Keep only essential data on the Pi
  - May run into space issues during training
  
- **32GB:** ✅ **Recommended** - Provides comfortable headroom for:
  - Multiple model checkpoints
  - Training logs
  - Temporary files
  - Future updates

**If the bundle with 16GB is significantly cheaper, it can work, but 32GB is safer for a smooth experience.**

## Training Approach

**Sequential Client Training:**
- We use **1 Raspberry Pi** instead of 2
- Clients run **sequentially** (not simultaneously)
- The Pi alternates between being Client 1 and Client 2 in each federated learning round
- This approach:
  - Reduces hardware costs by 50% (from ~8,272 TL to ~4,136 TL)
  - Still demonstrates federated learning effectively
  - Maintains experimental validity for IID vs Non-IID comparison

## Where to Buy

- **Robotistan:** Raspberry Pi 5 and power supply (check for bundles with MicroSD card)
- **Amazon:** MicroSD card (or any electronics retailer)

## Setup Timeline

- **Week 1:** Purchase and set up Raspberry Pi 5
- **Week 1:** Install Raspberry Pi OS and Python 3.11
- **Week 1:** Test network connectivity with laptop
- **Week 4:** Transfer datasets to Pi for federated training

