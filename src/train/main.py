"""
Main training script for Zhang et al. 2017 experiments.

Usage:
    python -m src.train.main <config.yaml>
    python -m src.train.main recipes/table1/inception_crop_no_wd_no.yaml
"""
import sys
import os
import json
import torch
import torch.nn as nn

from src.config import ExperimentConfig, get_optimizer, get_scheduler
from .utils import set_seed, get_model, get_dataloaders
from .display import print_run_card, get_experiment_id
from .experiment import check_experiment_completed, save_results
from .checkpoint import load_checkpoint
from .trainer import train_epoch, test


def main(config: ExperimentConfig):
    """Main training loop."""

    # Enforce wandb logging
    if not config.use_wandb:
        raise ValueError(
            "WandB logging is required but use_wandb=false in config.\n"
            "Please set 'use_wandb: true' in the 'logging:' section of your YAML config."
        )

    # Validate wandb settings
    if not config.wandb_project:
        raise ValueError("wandb_project must be set in config")
    if not config.wandb_entity:
        raise ValueError("wandb_entity must be set in config")

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

    if is_completed:
        print(f"\n{'='*60}")
        print(f"⚠️  EXPERIMENT ALREADY COMPLETED")
        print(f"{'='*60}")
        print(f"Results file: {existing_results}")
        print(f"\nTo re-run this experiment, delete:")
        print(f"  - Results file: {existing_results}")
        print(f"  - Checkpoint: {config.save_dir}/{experiment_id}_checkpoint.pth")
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

    # Initialize wandb
    import wandb

    # Create descriptive run name
    reg_parts = []
    if config.weight_decay > 0:
        reg_parts.append(f"wd{config.weight_decay}")
    if config.data.random_crop:
        reg_parts.append("crop")
    if config.data.augment_flip_rotate:
        reg_parts.append("aug")
    reg_suffix = "_" + "_".join(reg_parts) if reg_parts else "_baseline"

    wandb_run_name = f"{config.model_name}_{config.data.dataset}{reg_suffix}"

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
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': optimizer.param_groups[0]['lr'],
        })

        # Check early stopping condition
        early_stop = False
        if config.early_stopping_enabled and epoch >= config.early_stopping_min_epochs and len(all_metrics['test_acc']) >= config.early_stopping_window:
            # Get test accuracies for the last 'window' epochs
            recent_test_accs = all_metrics['test_acc'][-config.early_stopping_window:]
            acc_range = max(recent_test_accs) - min(recent_test_accs)

            if acc_range < 1.0:
                print(f"\n{'='*60}")
                print(f"⚠️  EARLY STOPPING TRIGGERED")
                print(f"{'='*60}")
                print(f"Minimum epochs reached: {epoch} >= {config.early_stopping_min_epochs}")
                print(f"Test accuracy range over last {config.early_stopping_window} epochs: {acc_range:.4f}% < 1.0%")
                print(f"Recent test accuracies: {[f'{acc:.2f}' for acc in recent_test_accs]}")
                print(f"Stopping training and saving model...")
                print(f"{'='*60}\n")
                early_stop = True

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

        # Break if early stopping triggered
        if early_stop:
            break

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

    # Log final summary to wandb
    wandb.summary['best_test_acc'] = best_test_acc
    wandb.summary['final_test_acc'] = all_metrics['final_test_acc']
    wandb.finish()


if __name__ == '__main__':
    # Require config file as first argument
    if len(sys.argv) != 2:
        print("Usage: python -m src.train.main <config.yaml>")
        print("\nExample:")
        print("  python -m src.train.main recipes/table1/inception_crop_no_wd_no.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Load config from YAML
    print(f"Loading configuration from: {config_path}")
    config = ExperimentConfig.from_yaml(config_path)

    # Run training
    main(config)
