"""
Training configuration and utilities for Zhang et al. 2017 experiments.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal
import yaml
from pathlib import Path
from src.dataset import DataConfig


@dataclass
class ExperimentConfig:
    """
    Configuration for Zhang et al. 2017 experiments.

    This config captures all the paper-critical knobs for reproducing
    their CIFAR-10 experiments.
    """

    # Model configuration
    model_name: Literal[
        'inception',
        'inception_no_bn',
        'small_alexnet',
        'mlp_1x512',
        'mlp_3x512'
    ] = 'inception'
    num_classes: int = 10
    input_shape: tuple = (3, 28, 28)  # (C, H, W) - CIFAR-10 is always cropped to 28×28

    # Data configuration (composed)
    data: DataConfig = field(default_factory=DataConfig)

    # Regularization (model training)
    weight_decay: float = 0.0  # L2 regularization coefficient

    # Training hyperparameters
    num_epochs: int = 100
    learning_rate: float = 0.1
    momentum: float = 0.9
    lr_schedule: Optional[str] = 'exponential'  # 'step', 'cosine', 'exponential', or None

    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_window: int = 6
    early_stopping_min_epochs: int = 25

    # Optimizer
    optimizer: Literal['sgd', 'adam'] = 'sgd'

    # Logging and checkpointing
    use_wandb: bool = False
    wandb_project: str = 'FOLI-Project'
    wandb_entity: Optional[str] = 'alirezasakhaeirad'
    save_dir: str = './checkpoints'
    log_interval: int = 10  # Log every N batches

    # Reproducibility
    seed: int = 42

    # Device
    device: str = 'cuda'  # 'cuda', 'cpu', or 'mps'

    @property
    def explicit_reg_off(self) -> bool:
        """
        Check if all explicit regularization is turned off.
        This is the setting used for baseline experiments.
        """
        return (
            self.weight_decay == 0.0 and
            not self.data.random_crop and
            not self.data.augment_flip_rotate
        )

    def get_model_weight_decay(self) -> float:
        """
        Get weight decay coefficient for the current model.

        The paper mentions "default coefficient for each model" (Table 4)
        but doesn't specify exact values. These are reasonable defaults
        based on common practice for these architectures.

        You can override by setting weight_decay directly.
        """
        if self.weight_decay > 0:
            return self.weight_decay

        # Default coefficients (can be tuned)
        defaults = {
            'inception': 0.0005,
            'inception_no_bn': 0.0005,
            'small_alexnet': 0.0005,
            'mlp_1x512': 0.0001,
            'mlp_3x512': 0.0001,
        }
        return defaults.get(self.model_name, 0.0)


    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ExperimentConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ExperimentConfig':
        """
        Create configuration from dictionary.

        Supports both flat and nested YAML structures:
        - Nested: sections like 'model:', 'data:', 'training:', etc.
        - Flat: all fields at top level (backward compatible)

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            ExperimentConfig instance
        """
        # Make a copy to avoid modifying original
        config = config_dict.copy()

        # Flatten nested sections into top-level
        sections_to_flatten = ['model', 'training', 'regularization', 'early_stopping', 'logging']
        for section in sections_to_flatten:
            if section in config and isinstance(config[section], dict):
                section_dict = config.pop(section)
                # Merge section contents into top-level (don't overwrite existing keys)
                for key, value in section_dict.items():
                    if key not in config:
                        config[key] = value

        # Handle data config
        data_dict = config.pop('data', {})

        # For backward compatibility: move data-related fields from top-level into data_dict
        data_fields = {f.name for f in DataConfig.__dataclass_fields__.values()}
        for key in list(config.keys()):
            if key in data_fields and key not in data_dict:
                data_dict[key] = config.pop(key)

        data_config = DataConfig.from_dict(data_dict)

        # Filter to only include valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config.items() if k in valid_fields and k != 'data'}

        # Handle tuple conversion for input_shape
        if 'input_shape' in filtered_dict and isinstance(filtered_dict['input_shape'], list):
            filtered_dict['input_shape'] = tuple(filtered_dict['input_shape'])

        return cls(data=data_config, **filtered_dict)

    def to_dict(self):
        """Convert config to dictionary for logging."""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'data': self.data.to_dict(),
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'lr_schedule': self.lr_schedule,
            'optimizer': self.optimizer,
            'seed': self.seed,
            'explicit_reg_off': self.explicit_reg_off,
            'early_stopping_enabled': self.early_stopping_enabled,
            'early_stopping_window': self.early_stopping_window,
            'early_stopping_min_epochs': self.early_stopping_min_epochs,
        }

    def __str__(self):
        return str(self.to_dict())

def get_optimizer(model, config: ExperimentConfig):
    """
    Get optimizer with paper-faithful configuration.

    Args:
        model: PyTorch model
        config: ExperimentConfig object

    Returns:
        PyTorch optimizer
    """
    import torch.optim as optim

    if config.optimizer == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_scheduler(optimizer, config: ExperimentConfig):
    """
    Get learning rate scheduler if specified.

    Args:
        optimizer: PyTorch optimizer
        config: ExperimentConfig object

    Returns:
        PyTorch scheduler or None
    """
    import torch.optim.lr_scheduler as lr_scheduler

    if config.lr_schedule is None:
        return None
    elif config.lr_schedule == 'step':
        # Step decay (common for CIFAR10)
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[60, 80],
            gamma=0.1
        )
    elif config.lr_schedule == 'cosine':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs
        )
    elif config.lr_schedule == 'exponential':
        # Decay by gamma every epoch
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")


# Preset configurations for common experiments

def get_baseline_config(**kwargs) -> ExperimentConfig:
    """
    Baseline configuration: no explicit regularization.
    This is the "Inception" row in Table 1.
    - Center crop to 28×28 (no random crop)
    - No weight decay
    - No augmentation
    """
    data_config = DataConfig(
        random_crop=False,
        augment_flip_rotate=False,
    )
    config = ExperimentConfig(
        model_name='inception',
        data=data_config,
        weight_decay=0.0,
    )
    # Override with any user-provided kwargs
    for key, value in kwargs.items():
        if key in ['random_crop', 'augment_flip_rotate', 'batch_size', 'num_workers']:
            setattr(config.data, key, value)
        else:
            setattr(config, key, value)
    return config


def get_regularized_config(**kwargs) -> ExperimentConfig:
    """
    Configuration with explicit regularization enabled (Table 1).
    - Random crop to 28×28 (data augmentation)
    - Weight decay enabled
    """
    data_config = DataConfig(
        random_crop=True,  # Enable random crop (center crop → random crop)
        augment_flip_rotate=False,
    )
    config = ExperimentConfig(
        model_name='inception',
        data=data_config,
        weight_decay=0.0005,  # Enable weight decay
    )
    for key, value in kwargs.items():
        if key in ['random_crop', 'augment_flip_rotate', 'batch_size', 'num_workers']:
            setattr(config.data, key, value)
        else:
            setattr(config, key, value)
    return config
