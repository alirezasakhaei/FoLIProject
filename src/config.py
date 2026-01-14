"""
Training configuration and utilities for Zhang et al. 2017 experiments.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal
import yaml
from pathlib import Path


@dataclass
class ExperimentConfig:
    """
    Configuration for Zhang et al. 2017 experiments.
    
    This config captures all the paper-critical knobs for reproducing
    their CIFAR10/MNIST experiments.
    """
    
    # Model configuration
    model_name: Literal[
        'small_inception',
        'small_inception_no_bn',
        'small_alexnet',
        'mlp_1x512',
        'mlp_3x512'
    ] = 'small_inception'
    num_classes: int = 10
    input_shape: tuple = (3, 32, 32)  # (C, H, W)
    
    # Dataset configuration
    dataset: Literal['cifar10', 'mnist'] = 'cifar10'
    data_root: str = './data'
    
    # Randomization experiments (Table 1, Appendix E)
    randomization: Optional[Literal[
        'random_labels',
        'partial_corrupt',
        'shuffled_pixels',
        'random_pixels',
        'gaussian_pixels'
    ]] = None
    corruption_prob: float = 0.0  # For partial_corrupt
    randomization_seed: Optional[int] = 42
    
    # Regularization toggles (Table 1, Table 4)
    weight_decay: float = 0.0  # L2 regularization coefficient
    random_crop: bool = False  # Table 1 "random crop" toggle
    augment_flip_rotate: bool = False  # Appendix E augmentation (flip + rotate 25°)
    
    # Preprocessing
    center_crop_size: int = 28  # CIFAR10 center crop to 28x28
    
    # Training hyperparameters
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 0.01
    momentum: float = 0.9
    lr_schedule: Optional[str] = None  # e.g., 'step', 'cosine'
    
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
    device: str = 'cuda'  # or 'cpu', 'mps'
    
    @property
    def explicit_reg_off(self) -> bool:
        """
        Check if all explicit regularization is turned off.
        This is the setting used for random label experiments.
        """
        return (
            self.weight_decay == 0.0 and
            not self.random_crop and
            not self.augment_flip_rotate
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
            'small_inception': 0.0005,
            'small_inception_no_bn': 0.0005,
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
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            ExperimentConfig instance
        """
        # Filter to only include valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        # Handle tuple conversion for input_shape
        if 'input_shape' in filtered_dict and isinstance(filtered_dict['input_shape'], list):
            filtered_dict['input_shape'] = tuple(filtered_dict['input_shape'])
        
        return cls(**filtered_dict)
    
    def to_dict(self):
        """Convert config to dictionary for logging."""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'dataset': self.dataset,
            'randomization': self.randomization,
            'corruption_prob': self.corruption_prob,
            'weight_decay': self.weight_decay,
            'random_crop': self.random_crop,
            'augment_flip_rotate': self.augment_flip_rotate,
            'center_crop_size': self.center_crop_size,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'optimizer': self.optimizer,
            'seed': self.seed,
            'explicit_reg_off': self.explicit_reg_off,
        }


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
    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")


# Preset configurations for common experiments

def get_baseline_config(**kwargs) -> ExperimentConfig:
    """
    Baseline configuration: no randomization, no explicit regularization.
    This is the "Inception" row in Table 1.
    """
    config = ExperimentConfig(
        model_name='small_inception',
        randomization=None,
        weight_decay=0.0,
        random_crop=False,
        augment_flip_rotate=False,
    )
    # Override with any user-provided kwargs
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def get_random_labels_config(**kwargs) -> ExperimentConfig:
    """
    Random labels experiment (Table 1, Appendix E).
    All explicit regularization off.
    """
    config = ExperimentConfig(
        model_name='small_inception',
        randomization='random_labels',
        weight_decay=0.0,
        random_crop=False,
        augment_flip_rotate=False,
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def get_regularized_config(**kwargs) -> ExperimentConfig:
    """
    Configuration with explicit regularization enabled.
    This is for testing the effect of regularization (Table 1).
    """
    config = ExperimentConfig(
        model_name='small_inception',
        randomization=None,
        weight_decay=0.0005,  # Enable weight decay
        random_crop=True,  # Enable random crop
        augment_flip_rotate=False,  # Typically not used together with random_crop
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def get_augmented_random_labels_config(**kwargs) -> ExperimentConfig:
    """
    Random labels with augmentation (Appendix E, Table 4).
    Tests if augmentation helps when fitting random labels.
    """
    config = ExperimentConfig(
        model_name='small_inception',
        randomization='random_labels',
        weight_decay=0.0,
        random_crop=False,
        augment_flip_rotate=True,  # Flip + rotate augmentation
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config
