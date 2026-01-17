"""
Data configuration for CIFAR-10 experiments.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class DataConfig:
    """
    Configuration for CIFAR-10 data loading and preprocessing.

    Following Zhang et al. 2017:
    - CIFAR-10 32×32 → always cropped to 28×28
    - random_crop controls: center crop (baseline) vs random crop (augmentation)
    - Per-image whitening applied after cropping
    """

    # Dataset
    dataset: Literal['cifar10', 'imagenet', 'cifar100'] = 'cifar10'
    data_root: str = './data'

    # Preprocessing - crop size depends on dataset
    # CIFAR-10: always 28×28 (from 32×32)
    # ImageNet: typically 224×224 (from 256×256 or variable)
    crop_size: int = 28  # Output size after cropping

    # Augmentation toggles (Table 1, Appendix E)
    random_crop: bool = False  # If False: center crop, If True: random crop
    augment_flip_rotate: bool = False  # Random flip + rotation ≤25°

    # DataLoader settings
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True

    # Randomization
    randomization: Literal['random_labels', 'corrupt_labels', 'shuffled_pixels', 'random_pixels', 'gaussian', 'none'] = 'none'
    corruption_prob: float = 0.0

    # Data fraction (for using subset of data)
    data_fraction: float = 1.0  # Fraction of data to use (0.0 to 1.0)

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if self.dataset == 'cifar10' and self.crop_size != 28:
            raise ValueError(f"CIFAR-10 must use crop_size=28, got {self.crop_size}")
        if self.dataset == 'imagenet' and self.crop_size not in [224, 299]:
            raise ValueError(f"ImageNet typically uses crop_size=224 or 299, got {self.crop_size}")
        if not 0.0 < self.data_fraction <= 1.0:
            raise ValueError(f"data_fraction must be in (0.0, 1.0], got {self.data_fraction}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'DataConfig':
        """Create DataConfig from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'dataset': self.dataset,
            'data_root': self.data_root,
            'crop_size': self.crop_size,
            'random_crop': self.random_crop,
            'augment_flip_rotate': self.augment_flip_rotate,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'randomization': self.randomization,
            'corruption_prob': self.corruption_prob,
            'data_fraction': self.data_fraction,
            'seed': self.seed,
        }
