import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import argparse
import json

# Configuration
MODELS_DIR = Path("models/federated")
RESULTS_DIR = Path("results/federated")
NUM_ROUNDS = 5  # Number of federated learning rounds
MIN_CLIENTS = 2  # Minimum number of clients to start training
MIN_AVAILABLE_CLIENTS = 2  # Minimum clients that must be available

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Class mapping
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'divide', 'equals', 'minus', 'multiply', 'plus']


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


def get_initial_parameters():
    """Initialize model parameters"""
    model = ASLModel(num_classes=len(CLASSES), pretrained=True)
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from multiple clients using weighted average"""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def save_global_model(parameters, round_num, timestamp):
    """Save the global model parameters"""
    model = ASLModel(num_classes=len(CLASSES), pretrained=False)
    
    # Convert parameters to state dict
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    
    # Create checkpoint
    checkpoint = {
        'round': round_num,
        'model_state_dict': state_dict,
        'classes': CLASSES,
        'num_classes': len(CLASSES),
        'timestamp': timestamp
    }
    
    # Save checkpoint
    save_path = MODELS_DIR / f"federated_model_round_{round_num}_{timestamp}.pth"
    torch.save(checkpoint, save_path)
    print(f"✓ Global model saved: {save_path.name}")
    
    return save_path


class FederatedStrategy(FedAvg):
    """Custom federated learning strategy with logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_accuracies = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_parameters = None
        self.best_accuracy = 0.0
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results"""
        print(f"\n{'='*80}")
        print(f"Round {server_round}/{NUM_ROUNDS} - Aggregating Training Results")
        print(f"{'='*80}")
        print(f"✓ Received updates from {len(results)} clients")
        if failures:
            print(f"⚠ {len(failures)} clients failed")
        
        # Call parent to aggregate
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Save parameters for later use
        if aggregated_parameters is not None:
            self.current_parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
        
        # Save global model every 5 rounds
        if server_round % 5 == 0 and aggregated_parameters is not None:
            save_global_model(
                self.current_parameters,
                server_round,
                self.timestamp
            )
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results"""
        print(f"\n{'='*80}")
        print(f"Round {server_round}/{NUM_ROUNDS} - Aggregating Evaluation Results")
        print(f"{'='*80}")
        
        if not results:
            print("⚠ No evaluation results received")
            return None, {}
        
        # Call parent to aggregate
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Get accuracy
        if "accuracy" in metrics_aggregated:
            accuracy = metrics_aggregated["accuracy"]
            self.round_accuracies.append({
                'round': server_round,
                'accuracy': accuracy,
                'loss': loss_aggregated
            })
            
            print(f"✓ Global Model Performance:")
            print(f"  Loss: {loss_aggregated:.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            
            # Save best model
            if accuracy > self.best_accuracy and self.current_parameters is not None:
                self.best_accuracy = accuracy
                save_global_model(
                    self.current_parameters,
                    server_round,
                    self.timestamp
                )
                # Also save as "best"
                best_path = MODELS_DIR / f"federated_model_best_{self.timestamp}.pth"
                model = ASLModel(num_classes=len(CLASSES), pretrained=False)
                params_dict = zip(model.state_dict().keys(), self.current_parameters)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                checkpoint = {
                    'round': server_round,
                    'model_state_dict': state_dict,
                    'classes': CLASSES,
                    'num_classes': len(CLASSES),
                    'timestamp': self.timestamp,
                    'best_accuracy': self.best_accuracy
                }
                torch.save(checkpoint, best_path)
                print(f"★ New best accuracy ({accuracy:.2f}%)! Model saved: {best_path.name}")
        
        print(f"{'='*80}\n")
        return loss_aggregated, metrics_aggregated
    
    def finalize(self):
        """Save final results"""
        if self.round_accuracies:
            results_file = RESULTS_DIR / f"federated_training_results_{self.timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'rounds': NUM_ROUNDS,
                    'min_clients': MIN_CLIENTS,
                    'history': self.round_accuracies,
                    'best_accuracy': max(self.round_accuracies, key=lambda x: x['accuracy'])
                }, f, indent=4)
            print(f"✓ Training results saved: {results_file.name}")


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Server for ASL Recognition')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host address (0.0.0.0 to accept all connections)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Server port (default: 8080)')
    parser.add_argument('--rounds', type=int, default=5,
                       help='Number of federated learning rounds')
    parser.add_argument('--min-clients', type=int, default=2,
                       help='Minimum number of clients required per round')
    args = parser.parse_args()
    
    global NUM_ROUNDS, MIN_CLIENTS, MIN_AVAILABLE_CLIENTS
    NUM_ROUNDS = args.rounds
    MIN_CLIENTS = args.min_clients
    MIN_AVAILABLE_CLIENTS = args.min_clients
    
    print("=" * 80)
    print("Federated Learning Server - ASL Gesture Recognition")
    print("=" * 80)
    print(f"Server address: {args.host}:{args.port}")
    print(f"Number of rounds: {NUM_ROUNDS}")
    print(f"Minimum clients per round: {MIN_CLIENTS}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print()
    
    # Get initial model parameters
    print("Initializing global model...")
    initial_parameters = get_initial_parameters()
    print(f"✓ Model initialized with {len(initial_parameters)} parameter tensors")
    print()
    
    # Create strategy
    print("Creating federated learning strategy...")
    strategy = FederatedStrategy(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
    )
    print("✓ Strategy configured: FedAvg with weighted averaging")
    print()
    
    print("=" * 80)
    print("Starting Federated Learning Server")
    print("=" * 80)
    print(f"Waiting for at least {MIN_AVAILABLE_CLIENTS} clients to connect...")
    print()
    print("Instructions for clients:")
    print(f"  python src/client.py --server {get_server_address(args.host)}:{args.port}")
    print()
    print("=" * 80)
    print()
    
    # Start server
    try:
        fl.server.start_server(
            server_address=f"{args.host}:{args.port}",
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
        )
        
        # Finalize and save results
        print("\n" + "=" * 80)
        print("Federated Learning Completed!")
        print("=" * 80)
        strategy.finalize()
        
        if strategy.round_accuracies:
            best_round = max(strategy.round_accuracies, key=lambda x: x['accuracy'])
            print(f"\nBest Performance:")
            print(f"  Round: {best_round['round']}")
            print(f"  Accuracy: {best_round['accuracy']:.2f}%")
            print(f"  Loss: {best_round['loss']:.4f}")
        
        print("\n" + "=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nServer interrupted by user")
        strategy.finalize()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Is port {args.port} already in use?")
        print(f"  2. Check firewall settings")
        print(f"  3. Ensure clients can reach {args.host}:{args.port}")


def get_server_address(host):
    """Get appropriate server address for client instructions"""
    if host == "0.0.0.0":
        # Get local IP
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "localhost"
    return host


if __name__ == "__main__":
    main()

