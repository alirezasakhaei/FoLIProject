"""
Checkpoint management for saving and loading model states.
"""
import os
import torch


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
