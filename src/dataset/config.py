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
    dataset: Literal['cifar10'] = 'cifar10'
    data_root: str = './data'

    # Preprocessing - CIFAR-10 is ALWAYS cropped to 28×28
    crop_size: int = 28  # Output size after cropping (28×28 from 32×32)

    # Augmentation toggles (Table 1, Appendix E)
    random_crop: bool = False  # If False: center crop, If True: random crop
    augment_flip_rotate: bool = False  # Random flip + rotation ≤25°

    # DataLoader settings
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.dataset != 'cifar10':
            raise ValueError(f"Only 'cifar10' is supported, got '{self.dataset}'")
        if self.crop_size != 28:
            raise ValueError(f"CIFAR-10 must use crop_size=28, got {self.crop_size}")

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
        }
