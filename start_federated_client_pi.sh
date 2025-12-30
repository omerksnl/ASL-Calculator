#!/bin/bash
echo "============================================================"
echo "Federated Learning Client Starter (Raspberry Pi)"
echo "============================================================"
echo ""

# Get configuration from user
read -p "Enter server IP address (e.g., 192.168.1.100): " SERVER_IP
if [ -z "$SERVER_IP" ]; then
    echo "Error: Server IP is required!"
    exit 1
fi

echo ""
echo "Select experiment type:"
echo "  1) IID (balanced data)"
echo "  2) Non-IID (skewed data - operators)"
read -p "Choice [1-2]: " EXPERIMENT

# Pi is always Client 2
CLIENT=2

# Set data directory
if [ "$EXPERIMENT" == "1" ]; then
    EXP_TYPE="master_iid"
    EXP_NAME="IID"
else
    EXP_TYPE="master_noniid"
    EXP_NAME="Non-IID (operators)"
fi

DATA_DIR="data/master_dataset/${EXP_TYPE}/client${CLIENT}"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "venv311" ]; then
    source venv311/bin/activate
fi

echo ""
echo "============================================================"
echo "Configuration:"
echo "  Server:     $SERVER_IP:8080"
echo "  Experiment: $EXP_NAME"
echo "  Client:     Client 2 (Pi)"
echo "  Data:       $DATA_DIR"
echo "============================================================"
echo ""
echo "Starting client..."
echo ""

# Start client
python3 src/client.py --server $SERVER_IP:8080 --data-dir $DATA_DIR

