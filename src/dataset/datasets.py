"""
CIFAR-10 dataset factory for Zhang et al. 2017 experiments.

Following the paper's preprocessing:
- CIFAR-10 32×32 → cropped to 28×28 (center crop or random crop)
- Pixel values scaled to [0, 1] by ToTensor
- Per-image whitening (subtract mean, divide by adjusted stddev)
"""
from torch.utils.data import DataLoader
from torchvision import datasets
from .config import DataConfig
from .transforms import get_cifar10_transforms


class DatasetFactory:
    """
    Factory for creating CIFAR-10 train and test datasets with proper preprocessing.
    """

    @staticmethod
    def create_datasets(config: DataConfig):
        """
        Create train and test datasets based on config.

        Args:
            config: DataConfig instance with preprocessing settings

        Returns:
            tuple: (train_dataset, test_dataset)
        """
        # Training dataset transforms
        train_transform = get_cifar10_transforms(
            random_crop=config.random_crop,
            augment_flip_rotate=config.augment_flip_rotate,
            center_crop_size=config.crop_size,
        )

        # Test dataset transforms (always center crop, no augmentation)
        test_transform = get_cifar10_transforms(
            random_crop=False,
            augment_flip_rotate=False,
            center_crop_size=config.crop_size,
        )

        # Create datasets
        train_dataset = datasets.CIFAR10(
            root=config.data_root,
            train=True,
            transform=train_transform,
            download=True,
        )

        test_dataset = datasets.CIFAR10(
            root=config.data_root,
            train=False,
            transform=test_transform,
            download=True,
        )

        return train_dataset, test_dataset

    @staticmethod
    def create_dataloaders(config: DataConfig):
        """
        Create train and test dataloaders based on config.

        Args:
            config: DataConfig instance with preprocessing and loader settings

        Returns:
            tuple: (train_loader, test_loader)
        """
        train_dataset, test_dataset = DatasetFactory.create_datasets(config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        return train_loader, test_loader

