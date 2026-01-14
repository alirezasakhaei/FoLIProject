"""
Utility functions for training.
"""
import random
import numpy as np
import torch
from src.config import ExperimentConfig
from src.dataset import DatasetFactory
from src.models import (
    inception,
    inception_no_bn,
    small_alexnet,
    mlp_1x512,
    mlp_3x512,
)

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

    # CIFAR-10 is always 28×28 after cropping
    return model_fn(num_classes=config.num_classes, input_shape=config.input_shape)


def get_dataloaders(config: ExperimentConfig):
    """
    Get train and test dataloaders using the DatasetFactory.

    Args:
        config: ExperimentConfig with nested DataConfig

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Use the factory to create dataloaders from the data config
    return DatasetFactory.create_dataloaders(config.data)


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
