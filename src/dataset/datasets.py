"""
CIFAR-10 dataset factory for Zhang et al. 2017 experiments.

Following the paper's preprocessing:
- CIFAR-10 32×32 → cropped to 28×28 (center crop or random crop)
- Pixel values scaled to [0, 1] by ToTensor
- Per-image whitening (subtract mean, divide by adjusted stddev)
"""
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from .config import DataConfig
from .transforms import get_cifar10_transforms, get_imagenet_transforms, ShuffledPixels, RandomPixels, GaussianPixels
import torch
import numpy as np
import os
from .utils import num_class_counts

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
        # Prepare pixel randomization transforms if needed
        train_pixel_randomization = None
        test_pixel_randomization = None

        if config.randomization == 'shuffled_pixels':
            # Create a fixed permutation for all images (same for train and test)
            if config.dataset in ['cifar10', 'cifar100']:
                # CIFAR images are 32x32x3 = 3072 pixels
                np.random.seed(config.seed if hasattr(config, 'seed') else 42)
                permutation = np.random.permutation(32 * 32)
            elif config.dataset == 'imagenet':
                # ImageNet images vary, permutation applied after resize
                np.random.seed(config.seed if hasattr(config, 'seed') else 42)
                permutation = np.random.permutation(256 * 256)
            train_pixel_randomization = ShuffledPixels(permutation)
            test_pixel_randomization = ShuffledPixels(permutation)
        elif config.randomization == 'random_pixels':
            # Each image gets a different random permutation (train only, test unchanged)
            train_pixel_randomization = RandomPixels()
            test_pixel_randomization = None  # Test set unchanged
        elif config.randomization == 'gaussian':
            # Replace pixels with Gaussian noise
            train_pixel_randomization = GaussianPixels()
            test_pixel_randomization = GaussianPixels()

        if config.dataset == 'cifar10':
            # Training dataset transforms
            train_transform = get_cifar10_transforms(
                random_crop=config.random_crop,
                augment_flip_rotate=config.augment_flip_rotate,
                center_crop_size=config.crop_size,
                pixel_randomization=train_pixel_randomization,
            )

            # Test dataset transforms (always center crop, no augmentation)
            test_transform = get_cifar10_transforms(
                random_crop=False,
                augment_flip_rotate=False,
                center_crop_size=config.crop_size,
                pixel_randomization=test_pixel_randomization,
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

        elif config.dataset == 'cifar100':
            # CIFAR-100 uses the same transforms as CIFAR-10 (both 32x32)
            # Training dataset transforms
            train_transform = get_cifar10_transforms(
                random_crop=config.random_crop,
                augment_flip_rotate=config.augment_flip_rotate,
                center_crop_size=config.crop_size,
                pixel_randomization=train_pixel_randomization,
            )

            # Test dataset transforms (always center crop, no augmentation)
            test_transform = get_cifar10_transforms(
                random_crop=False,
                augment_flip_rotate=False,
                center_crop_size=config.crop_size,
                pixel_randomization=test_pixel_randomization,
            )

            # Create datasets
            train_dataset = datasets.CIFAR100(
                root=config.data_root,
                train=True,
                transform=train_transform,
                download=True,
            )

            test_dataset = datasets.CIFAR100(
                root=config.data_root,
                train=False,
                transform=test_transform,
                download=True,
            )

        elif config.dataset == 'imagenet':
            # Training dataset transforms
            train_transform = get_imagenet_transforms(
                random_crop=config.random_crop,
                augment_flip_rotate=config.augment_flip_rotate,
                crop_size=config.crop_size,
                pixel_randomization=train_pixel_randomization,
            )

            # Test dataset transforms (always center crop, no augmentation)
            test_transform = get_imagenet_transforms(
                random_crop=False,
                augment_flip_rotate=False,
                crop_size=config.crop_size,
                pixel_randomization=test_pixel_randomization,
            )

            # Use ImageFolder for more flexibility with ImageNet directory structure
            # Expected structure: data_root/train/<class_folders> and data_root/val/<class_folders>
            train_dir = os.path.join(config.data_root, 'train')
            val_dir = os.path.join(config.data_root, 'val')

            if not os.path.exists(train_dir):
                raise RuntimeError(
                    f"ImageNet train directory not found: {train_dir}\n"
                    f"Please organize ImageNet data as:\n"
                    f"  {config.data_root}/train/<class_folders>/\n"
                    f"  {config.data_root}/val/<class_folders>/"
                )

            if not os.path.exists(val_dir):
                raise RuntimeError(
                    f"ImageNet val directory not found: {val_dir}\n"
                    f"Please organize ImageNet data as:\n"
                    f"  {config.data_root}/train/<class_folders>/\n"
                    f"  {config.data_root}/val/<class_folders>/"
                )

            # Create datasets using ImageFolder
            train_dataset = datasets.ImageFolder(
                root=train_dir,
                transform=train_transform,
            )

            test_dataset = datasets.ImageFolder(
                root=val_dir,
                transform=test_transform,
            )

        # Apply label randomization if needed
        # Note: For ImageFolder, targets is a list, so we convert to tensor
        if config.randomization == 'random_labels':
            seed = config.seed if hasattr(config, 'seed') else 42
            # Set seed for reproducibility
            generator = torch.Generator().manual_seed(seed)

            if hasattr(train_dataset, 'targets'):
                num_samples = len(train_dataset.targets)
                train_dataset.targets = torch.randint(
                    0, num_class_counts(config.dataset),
                    (num_samples,),
                    generator=generator
                ).tolist()  # Convert back to list for ImageFolder compatibility
            elif hasattr(train_dataset, 'samples'):
                # ImageFolder also has samples attribute
                num_samples = len(train_dataset.samples)
                random_labels = torch.randint(
                    0, num_class_counts(config.dataset),
                    (num_samples,),
                    generator=generator
                ).tolist()
                # Update both samples and targets
                train_dataset.samples = [(path, label) for (path, _), label in zip(train_dataset.samples, random_labels)]
                train_dataset.targets = random_labels

        elif config.randomization == 'corrupt_labels':
            seed = config.seed if hasattr(config, 'seed') else 42
            torch.manual_seed(seed)

            # with probability config.corruption_prob, corrupt each label with a random label
            if hasattr(train_dataset, 'targets'):
                targets_list = train_dataset.targets if isinstance(train_dataset.targets, list) else train_dataset.targets.tolist()
                targets_tensor = torch.tensor(targets_list)
                random_vector_to_corrupt = torch.rand(len(targets_tensor)) < config.corruption_prob
                num_to_corrupt = random_vector_to_corrupt.sum().item()
                if num_to_corrupt > 0:
                    targets_tensor[random_vector_to_corrupt] = torch.randint(
                        0, num_class_counts(config.dataset),
                        (num_to_corrupt,)
                    )
                train_dataset.targets = targets_tensor.tolist()

                # Update samples if it's an ImageFolder
                if hasattr(train_dataset, 'samples'):
                    train_dataset.samples = [(path, label) for (path, _), label in zip(train_dataset.samples, train_dataset.targets)]

        # Apply data fraction if needed
        if config.data_fraction < 1.0:
            np.random.seed(config.seed if hasattr(config, 'seed') else 42)
            train_size = len(train_dataset)
            indices = np.random.permutation(train_size)
            subset_size = int(train_size * config.data_fraction)
            train_indices = indices[:subset_size]
            train_dataset = Subset(train_dataset, train_indices)

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

