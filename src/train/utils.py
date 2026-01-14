"""
Utility functions for training.
"""
import random
import numpy as np
import torch


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
