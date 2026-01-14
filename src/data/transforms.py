"""
Data transformations for Zhang et al. 2017 experiments.

Implements the paper's preprocessing pipeline:
- CIFAR10: divide by 255, center crop to 28x28, per-image whitening
- Optional augmentations: random crop, flip, rotation
"""
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np


class PerImageWhitening:
    """
    Per-image whitening matching TensorFlow's per_image_standardization.
    
    TensorFlow formula: (x - mean) / adjusted_stddev
    where adjusted_stddev = max(stddev, 1.0 / sqrt(num_elements))
    
    This matches the exact behavior from Zhang et al. 2017's TensorFlow implementation.
    """
    def __call__(self, tensor):
        """
        Args:
            tensor: PyTorch tensor of shape (C, H, W)
        Returns:
            Whitened tensor matching TF per_image_standardization
        """
        # Compute mean and std over all elements in the image
        mean = tensor.mean()
        # Use unbiased=False to match TensorFlow's population std
        std = tensor.std(unbiased=False)
        
        # TensorFlow's adjusted stddev to prevent division by zero
        # adjusted_stddev = max(stddev, 1.0 / sqrt(num_elements))
        num_elements = tensor.numel()
        adjusted_stddev = max(std.item(), 1.0 / (num_elements ** 0.5))
        
        return (tensor - mean) / adjusted_stddev



class ToFloat:
    """Converts tensor to float and divides by 255."""
    def __call__(self, tensor):
        return tensor.float() / 255.0


def get_cifar10_transforms(
    random_crop=False,
    augment_flip_rotate=False,
    center_crop_size=28,
):
    """
    Get CIFAR10 transforms following Zhang et al. 2017.
    
    Base preprocessing (always applied):
    - Convert to tensor
    - Divide by 255 to [0, 1]
    - Center crop to 28x28
    - Per-image whitening
    
    Args:
        random_crop: If True, apply random crop instead of center crop (Table 1 toggle)
        augment_flip_rotate: If True, apply random flip + rotation up to 25° (Appendix E)
        center_crop_size: Size for center crop (default 28 for CIFAR10)
    
    Returns:
        transforms.Compose object
    """
    transform_list = []
    
    # Convert to tensor first
    transform_list.append(transforms.ToTensor())  # This gives us (C, H, W) in [0, 1]
    
    # Augmentation (if enabled)
    if augment_flip_rotate:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomRotation(degrees=25))
    
    # Cropping
    if random_crop:
        # Random crop to target size (for training with augmentation)
        transform_list.append(transforms.RandomCrop(center_crop_size))
    else:
        # Center crop (default for paper experiments)
        transform_list.append(transforms.CenterCrop(center_crop_size))
    
    # Per-image whitening (subtract mean, divide by std)
    transform_list.append(PerImageWhitening())
    
    return transforms.Compose(transform_list)


def get_mnist_transforms(
    random_crop=False,
    augment_flip_rotate=False,
):
    """
    Get MNIST transforms following Zhang et al. 2017.
    
    Base preprocessing:
    - Convert to tensor
    - Divide by 255 to [0, 1]
    - Per-image whitening
    
    Args:
        random_crop: If True, apply random crop
        augment_flip_rotate: If True, apply random flip + rotation
    
    Returns:
        transforms.Compose object
    """
    transform_list = []
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Augmentation (if enabled)
    if augment_flip_rotate:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomRotation(degrees=25))
    
    if random_crop:
        # For MNIST (28x28), could do padding + random crop
        transform_list.append(transforms.RandomCrop(28, padding=4))
    
    # Per-image whitening
    transform_list.append(PerImageWhitening())
    
    return transforms.Compose(transform_list)


def get_test_transforms(dataset='cifar10', center_crop_size=28):
    """
    Get test/validation transforms (no augmentation).
    
    Args:
        dataset: 'cifar10' or 'mnist'
        center_crop_size: Size for center crop (only for CIFAR10)
    
    Returns:
        transforms.Compose object
    """
    if dataset == 'cifar10':
        return get_cifar10_transforms(
            random_crop=False,
            augment_flip_rotate=False,
            center_crop_size=center_crop_size,
        )
    elif dataset == 'mnist':
        return get_mnist_transforms(
            random_crop=False,
            augment_flip_rotate=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
