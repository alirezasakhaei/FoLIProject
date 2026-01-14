"""
Experiment tracking and results management.
"""
import os
import json
from datetime import datetime
from src.config import ExperimentConfig
from .display import get_experiment_id


def check_experiment_completed(config: ExperimentConfig, results_dir: str = './results') -> tuple:
    """
    Check if experiment has already been completed.

    Returns:
        (completed: bool, results_path: str or None)
    """
    if not os.path.exists(results_dir):
        return False, None

    experiment_id = get_experiment_id(config)

    # Look for completed results file
    for filename in os.listdir(results_dir):
        if filename.startswith(experiment_id) and filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    results = json.load(f)
                    # Check if experiment completed all epochs
                    if 'metrics' in results and 'epochs' in results['metrics']:
                        completed_epochs = len(results['metrics']['epochs'])
                        target_epochs = results['config'].get('num_epochs', config.num_epochs)
                        if completed_epochs >= target_epochs:
                            return True, filepath
            except (json.JSONDecodeError, KeyError):
                continue

    return False, None


def save_results(results_dir, config, metrics):
    """Save experiment results to JSON file."""
    os.makedirs(results_dir, exist_ok=True)

    # Create descriptive filename using experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = get_experiment_id(config)
    filename = f"{experiment_id}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    # Combine config and metrics with completion status
    results = {
        'experiment_id': experiment_id,
        'config': config.to_dict(),
        'metrics': metrics,
        'timestamp': timestamp,
        'completed': True,
        'total_epochs': len(metrics.get('epochs', [])),
        'target_epochs': config.num_epochs,
    }

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {filepath}")
    return filepath
