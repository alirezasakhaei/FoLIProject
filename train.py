"""
Training script for Zhang et al. 2017 experiments.

Usage:
    python train.py --model inception --dataset cifar10
    python train.py --model inception --randomization random_labels
    python train.py --model inception --weight_decay 0.0005 --random_crop
"""
import argparse
import os
import random
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import our modules
from src.config import ExperimentConfig, get_optimizer, get_scheduler
from src.data import get_cifar10_transforms, get_mnist_transforms
from src.data import get_cifar10_dataset, get_mnist_dataset
from src.models import (
    inception,
    inception_no_bn,
    small_alexnet,
    mlp_1x512,
    mlp_3x512,
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_params_info(model):
    """
    Get detailed parameter count information for a model.
    
    Returns:
        dict with total_params, paper_params (excl. BN), trainable_params, non_trainable_params
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Paper convention: count Conv/FC params but EXCLUDE BatchNorm params (matches Table 1)
    paper_params = 0
    for name, p in model.named_parameters():
        if ".bn." not in name and "bn." not in name:
            paper_params += p.numel()
    
    return {
        'total_params': total_params,
        'paper_params': paper_params,  # Excludes BatchNorm (Table 1 convention)
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
    }


def print_run_card(config: ExperimentConfig, model, train_loader, test_loader, optimizer, scheduler, device):
    """
    Print a detailed run card with all experiment information.
    """
    params_info = get_model_params_info(model)
    
    # Get scheduler info
    scheduler_info = "None"
    if scheduler is not None:
        scheduler_type = type(scheduler).__name__
        if hasattr(scheduler, 'milestones'):
            scheduler_info = f"{scheduler_type} (milestones={list(scheduler.milestones)}, gamma={scheduler.gamma})"
        elif hasattr(scheduler, 'T_max'):
            scheduler_info = f"{scheduler_type} (T_max={scheduler.T_max})"
        elif hasattr(scheduler, 'gamma'):
            scheduler_info = f"{scheduler_type} (gamma={scheduler.gamma})"
        else:
            scheduler_info = scheduler_type
    
    # Calculate total training steps
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.num_epochs
    
    # Estimate samples seen (accounting for last batch potentially being smaller)
    approx_samples_per_epoch = len(train_loader.dataset)
    total_samples = approx_samples_per_epoch * config.num_epochs
    
    print("\n" + "="*80)
    print("RUN CARD - EXPERIMENT CONFIGURATION")
    print("="*80)
    
    print("\n┌─ EXPERIMENT METADATA ─────────────────────────────────────────────────────┐")
    print(f"│ Experiment ID        : {get_experiment_id(config):<54} │")
    print(f"│ Timestamp            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<54} │")
    print(f"│ Device               : {str(device):<54} │")
    print(f"│ Random Seed          : {config.seed:<54} │")
    print("└───────────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ MODEL ARCHITECTURE ──────────────────────────────────────────────────────┐")
    print(f"│ Model Name           : {config.model_name:<54} │")
    print(f"│ Input Shape          : {str(config.input_shape):<54} │")
    print(f"│ Effective Shape      : ({config.input_shape[0]}, {config.center_crop_size}, {config.center_crop_size}){'':<39} │")
    print(f"│ Num Classes          : {config.num_classes:<54} │")
    print(f"│ Paper Params (no BN) : {params_info['paper_params']:,}{'':<54}"[:-54] + f"{params_info['paper_params']:>20,} │")
    print(f"│ Total Parameters     : {params_info['total_params']:,}{'':<54}"[:-54] + f"{params_info['total_params']:>20,} │")
    print(f"│ Trainable Params     : {params_info['trainable_params']:,}{'':<54}"[:-54] + f"{params_info['trainable_params']:>20,} │")
    print(f"│ Non-trainable Params : {params_info['non_trainable_params']:,}{'':<54}"[:-54] + f"{params_info['non_trainable_params']:>20,} │")
    print("└───────────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ DATASET & DATA LOADING ──────────────────────────────────────────────────┐")
    print(f"│ Dataset              : {config.dataset.upper():<54} │")
    print(f"│ Train Samples        : {len(train_loader.dataset):,}{'':<54}"[:-54] + f"{len(train_loader.dataset):>20,} │")
    print(f"│ Test Samples         : {len(test_loader.dataset):,}{'':<54}"[:-54] + f"{len(test_loader.dataset):>20,} │")
    print(f"│ Batch Size           : {config.batch_size:<54} │")
    print(f"│ Steps per Epoch      : {steps_per_epoch:,}{'':<54}"[:-54] + f"{steps_per_epoch:>20,} │")
    print(f"│ Center Crop Size     : {config.center_crop_size}x{config.center_crop_size}{'':<52} │")
    print("└───────────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ RANDOMIZATION & CORRUPTION ──────────────────────────────────────────────┐")
    randomization_display = config.randomization if config.randomization else "None (True Labels)"
    print(f"│ Randomization Type   : {randomization_display:<54} │")
    if config.randomization == 'partial_corrupt':
        print(f"│ Corruption Prob      : {config.corruption_prob:<54.2%} │")
    print(f"│ Randomization Seed   : {config.randomization_seed if config.randomization_seed else 'N/A':<54} │")
    print("└───────────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ REGULARIZATION TECHNIQUES ───────────────────────────────────────────────┐")
    print(f"│ Explicit Reg OFF     : {str(config.explicit_reg_off):<54} │")
    print(f"│ Weight Decay         : {config.weight_decay:<54} │")
    print(f"│ Random Crop          : {str(config.random_crop):<54} │")
    print(f"│ Augment (Flip+Rotate): {str(config.augment_flip_rotate):<54} │")
    print("└───────────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ OPTIMIZATION & TRAINING ─────────────────────────────────────────────────┐")
    print(f"│ Optimizer            : {config.optimizer.upper():<54} │")
    print(f"│ Learning Rate        : {config.learning_rate:<54} │")
    print(f"│ Momentum             : {config.momentum:<54} │")
    print(f"│ LR Scheduler         : {scheduler_info:<54} │")
    print(f"│ Num Epochs           : {config.num_epochs:<54} │")
    print(f"│ Total Training Steps : {total_steps:,}{'':<54}"[:-54] + f"{total_steps:>20,} │")
    print(f"│ Est. Samples Seen    : {total_samples:,}{'':<54}"[:-54] + f"{total_samples:>20,} │")
    print("└───────────────────────────────────────────────────────────────────────────┘")
    
    if config.use_wandb:
        print("\n┌─ LOGGING & TRACKING ──────────────────────────────────────────────────────┐")
        print(f"│ Weights & Biases     : Enabled{'':<49} │")
        print(f"│ W&B Project          : {config.wandb_project:<54} │")
        print(f"│ W&B Entity           : {config.wandb_entity if config.wandb_entity else 'N/A':<54} │")
        print(f"│ Save Directory       : {config.save_dir:<54} │")
        print(f"│ Log Interval         : Every {config.log_interval} batches{'':<41} │")
        print("└───────────────────────────────────────────────────────────────────────────┘")
    else:
        print("\n┌─ LOGGING & TRACKING ──────────────────────────────────────────────────────┐")
        print(f"│ Weights & Biases     : Disabled{'':<48} │")
        print(f"│ Save Directory       : {config.save_dir:<54} │")
        print(f"│ Log Interval         : Every {config.log_interval} batches{'':<41} │")
        print("└───────────────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")


def get_model(config: ExperimentConfig):
    """Get model based on config."""
    model_map = {
        'inception': inception,
        'inception_no_bn': inception_no_bn,
        'small_alexnet': small_alexnet,
        'mlp_1x512': mlp_1x512,
        'mlp_3x512': mlp_3x512,
    }
    
    model_fn = model_map[config.model_name]
    
    # Adjust input shape based on center_crop_size
    # CNNs work with any size, but MLPs need exact dimensions
    actual_input_shape = (config.input_shape[0], config.center_crop_size, config.center_crop_size)
    
    return model_fn(num_classes=config.num_classes, input_shape=actual_input_shape)


def get_dataloaders(config: ExperimentConfig):
    """Get train and test dataloaders."""
    
    # Get transforms
    if config.dataset == 'cifar10':
        train_transform = get_cifar10_transforms(
            random_crop=config.random_crop,
            augment_flip_rotate=config.augment_flip_rotate,
            center_crop_size=config.center_crop_size,
        )
        test_transform = get_cifar10_transforms(
            random_crop=False,
            augment_flip_rotate=False,
            center_crop_size=config.center_crop_size,
        )
        
        train_dataset = get_cifar10_dataset(
            root=config.data_root,
            train=True,
            transform=train_transform,
            download=True,
            randomization=config.randomization,
            randomization_seed=config.randomization_seed,
            corruption_prob=config.corruption_prob,
        )
        
        test_dataset = get_cifar10_dataset(
            root=config.data_root,
            train=False,
            transform=test_transform,
            download=True,
            randomization=None,  # Never randomize test set
        )
        
    elif config.dataset == 'mnist':
        train_transform = get_mnist_transforms(
            random_crop=config.random_crop,
            augment_flip_rotate=config.augment_flip_rotate,
        )
        test_transform = get_mnist_transforms(
            random_crop=False,
            augment_flip_rotate=False,
        )
        
        train_dataset = get_mnist_dataset(
            root=config.data_root,
            train=True,
            transform=train_transform,
            download=True,
            randomization=config.randomization,
            randomization_seed=config.randomization_seed,
            corruption_prob=config.corruption_prob,
        )
        
        test_dataset = get_mnist_dataset(
            root=config.data_root,
            train=False,
            transform=test_transform,
            download=True,
            randomization=None,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, test_loader


def get_experiment_id(config: ExperimentConfig) -> str:
    """Generate unique experiment identifier from config."""
    model_name = config.model_name
    dataset = config.dataset
    randomization = config.randomization or "baseline"
    
    # Add regularization suffix
    reg_parts = []
    if config.weight_decay > 0:
        reg_parts.append(f"wd{config.weight_decay}")
    if config.random_crop:
        reg_parts.append("crop")
    if config.augment_flip_rotate:
        reg_parts.append("aug")
    
    reg_suffix = "_" + "_".join(reg_parts) if reg_parts else ""
    
    # Add corruption probability for partial corruption
    if config.randomization == 'partial_corrupt' and config.corruption_prob > 0:
        reg_suffix += f"_corrupt{int(config.corruption_prob * 100)}"
    
    return f"{model_name}_{dataset}_{randomization}{reg_suffix}"


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


def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    """Load checkpoint if it exists."""
    if not os.path.exists(checkpoint_path):
        return 0, 0.0, {}  # start_epoch, best_acc, previous_metrics
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_acc = checkpoint.get('best_test_acc', 0.0)
    previous_metrics = checkpoint.get('all_metrics', {})
    
    print(f"Resuming from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    return start_epoch, best_acc, previous_metrics


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


def train_epoch(model, loader, criterion, optimizer, device, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % config.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} '
                  f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def test(model, loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(loader)
    accuracy = 100. * correct / total
    
    return test_loss, accuracy


def main(config: ExperimentConfig):
    """Main training loop."""
    
    # Set seed
    set_seed(config.seed)
    
    # Setup device
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif config.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if experiment already completed
    experiment_id = get_experiment_id(config)
    
    is_completed, existing_results = check_experiment_completed(config, results_dir)
    
    if is_completed and not getattr(config, 'force_rerun', False):
        print(f"\n{'='*60}")
        print(f"⚠️  EXPERIMENT ALREADY COMPLETED")
        print(f"{'='*60}")
        print(f"Results file: {existing_results}")
        print(f"\nTo re-run this experiment, either:")
        print(f"  1. Add --force flag: ./run_experiment.sh --config <recipe> --force")
        print(f"  2. Delete the results file: {existing_results}")
        print(f"  3. Delete the checkpoint: {config.save_dir}/{experiment_id}_checkpoint.pth")
        print(f"{'='*60}\n")
        
        # Load and display existing results
        with open(existing_results, 'r') as f:
            results = json.load(f)
            metrics = results['metrics']
            print(f"Previous Results:")
            print(f"  Best test accuracy: {metrics.get('best_test_acc', 0):.2f}%")
            print(f"  Final test accuracy: {metrics.get('final_test_acc', 0):.2f}%")
            print(f"  Total epochs: {len(metrics.get('epochs', []))}")
            print(f"  Completed: {results.get('timestamp', 'Unknown')}")
        
        return  # Exit without re-running
    elif is_completed:
        print(f"\n⚠️  Experiment already completed, but --force flag detected. Re-running...")
        print(f"Previous results: {existing_results}\n")
    
    # Initialize wandb if requested
    wandb_run_name = None
    if config.use_wandb:
        import wandb
        # Create descriptive run name
        randomization = config.randomization or "baseline"
        reg_parts = []
        if config.weight_decay > 0:
            reg_parts.append(f"wd{config.weight_decay}")
        if config.random_crop:
            reg_parts.append("crop")
        if config.augment_flip_rotate:
            reg_parts.append("aug")
        reg_suffix = "_" + "_".join(reg_parts) if reg_parts else ""
        
        wandb_run_name = f"{config.model_name}_{config.dataset}_{randomization}{reg_suffix}"
        
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=wandb_run_name,
            config=config.to_dict(),
        )
    
    # Get model
    model = get_model(config).to(device)
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(config)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Print detailed run card
    print_run_card(config, model, train_loader, test_loader, optimizer, scheduler, device)    
    
    # Try to load checkpoint (use experiment ID for unique naming)
    checkpoint_path = os.path.join(
        config.save_dir,
        f'{experiment_id}_checkpoint.pth'
    )
    start_epoch, best_test_acc, previous_metrics = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler
    )
    
    # Track metrics for all epochs (restore from checkpoint if resuming)
    if previous_metrics:
        all_metrics = previous_metrics
        print(f"✓ Restored {len(all_metrics.get('epochs', []))} previous epochs of metrics")
    else:
        all_metrics = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epochs': [],
        }
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Record metrics
        all_metrics['epochs'].append(epoch)
        all_metrics['train_loss'].append(train_loss)
        all_metrics['train_acc'].append(train_acc)
        all_metrics['test_loss'].append(test_loss)
        all_metrics['test_acc'].append(test_acc)
        
        # Print results
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # Log to wandb
        if config.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'lr': optimizer.param_groups[0]['lr'],
            })
        
        # Save checkpoint every epoch (include all metrics for resumption)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_test_acc': max(best_test_acc, test_acc),
            'config': config.to_dict(),
            'all_metrics': all_metrics,  # Save metrics for resumption
            'experiment_id': experiment_id,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_path = os.path.join(
                config.save_dir,
                f'{experiment_id}_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved: {best_test_acc:.2f}%")
    
    # Save final results
    all_metrics['best_test_acc'] = best_test_acc
    all_metrics['final_train_acc'] = all_metrics['train_acc'][-1]
    all_metrics['final_test_acc'] = all_metrics['test_acc'][-1]
    
    results_path = save_results(results_dir, config, all_metrics)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Final test accuracy: {all_metrics['final_test_acc']:.2f}%")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")
    
    if config.use_wandb:
        # Log final summary
        wandb.summary['best_test_acc'] = best_test_acc
        wandb.summary['final_test_acc'] = all_metrics['final_test_acc']
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zhang et al. 2017 Experiments')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (overrides other arguments)')
    
    # Model
    parser.add_argument('--model', type=str, default='inception',
                       choices=['inception', 'inception_no_bn', 
                               'small_alexnet', 'mlp_1x512', 'mlp_3x512'])
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist'])
    parser.add_argument('--data_root', type=str, default='./data')
    
    # Randomization
    parser.add_argument('--randomization', type=str, default=None,
                       choices=[None, 'random_labels', 'partial_corrupt',
                               'shuffled_pixels', 'random_pixels', 'gaussian_pixels'])
    parser.add_argument('--corruption_prob', type=float, default=0.0)
    parser.add_argument('--randomization_seed', type=int, default=42)
    
    # Regularization
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--random_crop', action='store_true')
    parser.add_argument('--augment_flip_rotate', action='store_true')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr_schedule', type=str, default=None, choices=[None, 'step', 'cosine', 'exponential'])
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='FOLI-Project')
    parser.add_argument('--wandb_entity', type=str, default='alirezasakhaeirad')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--force', action='store_true',
                       help='Force re-run even if experiment already completed')
    
    args = parser.parse_args()
    
    # Load config from YAML if provided, otherwise use CLI args
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = ExperimentConfig.from_yaml(args.config)
        print(f"DEBUG: Loaded LR from YAML: {config.learning_rate}")
        print(f"DEBUG: Loaded Schedule from YAML: {config.lr_schedule}")
        # Override settings from CLI if provided
        if args.use_wandb:
            config.use_wandb = True
        # Add force flag to config
        config.force_rerun = args.force
    else:
        # Create config from args
        config = ExperimentConfig(
            model_name=args.model,
            dataset=args.dataset,
            data_root=args.data_root,
            randomization=args.randomization,
            corruption_prob=args.corruption_prob,
            randomization_seed=args.randomization_seed,
            weight_decay=args.weight_decay,
            random_crop=args.random_crop,
            augment_flip_rotate=args.augment_flip_rotate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            momentum=args.momentum,
            optimizer=args.optimizer,
            lr_schedule=args.lr_schedule,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            save_dir=args.save_dir,
            log_interval=args.log_interval,
            seed=args.seed,
            device=args.device,
        )
    
    main(config)
