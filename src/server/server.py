import flwr as fl
from typing import List, Tuple
import numpy as np

def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}

def main():
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,  # Sample 30% of available clients for training
        fraction_evaluate=0.2,  # Sample 20% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 2 clients for training
        min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
        min_available_clients=2,  # Wait until at least 2 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()