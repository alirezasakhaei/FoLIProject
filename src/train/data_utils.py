"""
Data loading utilities for training.
"""
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
