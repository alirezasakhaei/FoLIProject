"""
Dataset wrappers for Zhang et al. 2017 randomization experiments.

Implements:
- Random labels (all labels replaced with random ones)
- Partially corrupted labels (probability p of random label)
- Shuffled pixels (same permutation for all images)
- Random pixels (different permutation per image)
- Gaussian pixels (matched mean/variance)
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms


class RandomLabelDataset(Dataset):
    """
    Wrapper that replaces all labels with random ones.
    
    Important: Random labels are fixed at initialization and remain
    consistent across epochs (as per paper's methodology).
    """
    def __init__(self, base_dataset, num_classes=10, seed=None):
        """
        Args:
            base_dataset: Original dataset
            num_classes: Number of classes for random labels
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        
        # Generate random labels once and store them
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        self.random_labels = rng.randint(0, num_classes, len(base_dataset))
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]  # Ignore original label
        return img, self.random_labels[idx]


class PartiallyCorruptedDataset(Dataset):
    """
    Wrapper that corrupts labels with probability p.
    
    Each label has probability p of being replaced with a random label.
    The corruption pattern is fixed at initialization.
    """
    def __init__(self, base_dataset, corruption_prob=0.0, num_classes=10, seed=None):
        """
        Args:
            base_dataset: Original dataset
            corruption_prob: Probability of corrupting each label (0 to 1)
            num_classes: Number of classes
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        # Determine which samples to corrupt
        n_samples = len(base_dataset)
        self.corrupt_mask = rng.random(n_samples) < corruption_prob
        
        # Generate random labels for corrupted samples
        self.random_labels = rng.randint(0, num_classes, n_samples)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, true_label = self.base_dataset[idx]
        
        if self.corrupt_mask[idx]:
            return img, self.random_labels[idx]
        else:
            return img, true_label


class ShuffledPixelsDataset(Dataset):
    """
    Wrapper that applies the same pixel permutation to all images.
    
    This creates a consistent "scrambling" of spatial structure while
    preserving the pixel values themselves.
    """
    def __init__(self, base_dataset, seed=None):
        """
        Args:
            base_dataset: Original dataset
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        
        # Get a sample image to determine dimensions
        sample_img, _ = base_dataset[0]
        if isinstance(sample_img, torch.Tensor):
            self.img_shape = sample_img.shape
        else:
            # If PIL image, convert to get shape
            sample_tensor = transforms.ToTensor()(sample_img)
            self.img_shape = sample_tensor.shape
        
        # Generate a single permutation for all images
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        total_pixels = np.prod(self.img_shape)
        self.permutation = rng.permutation(total_pixels)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Convert to tensor if needed
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        # Flatten, permute, reshape
        flat_img = img.view(-1)
        shuffled_flat = flat_img[self.permutation]
        shuffled_img = shuffled_flat.view(self.img_shape)
        
        return shuffled_img, label


class RandomPixelsDataset(Dataset):
    """
    Wrapper that applies a different random permutation to each image.
    
    This destroys all spatial structure while keeping pixel statistics.
    """
    def __init__(self, base_dataset, seed=None):
        """
        Args:
            base_dataset: Original dataset
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        
        # Get image shape
        sample_img, _ = base_dataset[0]
        if isinstance(sample_img, torch.Tensor):
            self.img_shape = sample_img.shape
        else:
            sample_tensor = transforms.ToTensor()(sample_img)
            self.img_shape = sample_tensor.shape
        
        self.total_pixels = np.prod(self.img_shape)
        
        # Store seed for reproducibility
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Convert to tensor if needed
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        # Generate a unique permutation for this image
        # Use idx as additional seed component for reproducibility
        local_rng = np.random.RandomState(self.rng.randint(0, 2**32) + idx)
        permutation = local_rng.permutation(self.total_pixels)
        
        # Flatten, permute, reshape
        flat_img = img.view(-1)
        shuffled_flat = flat_img[permutation]
        shuffled_img = shuffled_flat.view(self.img_shape)
        
        return shuffled_img, label


class GaussianPixelsDataset(Dataset):
    """
    Wrapper that replaces images with Gaussian noise matched to dataset statistics.
    
    Computes mean and variance of the original dataset and generates
    Gaussian noise with matching statistics.
    """
    def __init__(self, base_dataset, seed=None):
        """
        Args:
            base_dataset: Original dataset
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        
        # Get image shape
        sample_img, _ = base_dataset[0]
        if isinstance(sample_img, torch.Tensor):
            self.img_shape = sample_img.shape
        else:
            sample_tensor = transforms.ToTensor()(sample_img)
            self.img_shape = sample_tensor.shape
        
        # Compute dataset statistics (sample-based for efficiency)
        self._compute_statistics()
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random
    
    def _compute_statistics(self, max_samples=1000):
        """Compute mean and std of the dataset."""
        n_samples = min(len(self.base_dataset), max_samples)
        pixel_values = []
        
        for i in range(n_samples):
            img, _ = self.base_dataset[i]
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
            pixel_values.append(img.numpy().flatten())
        
        all_pixels = np.concatenate(pixel_values)
        self.mean = np.mean(all_pixels)
        self.std = np.std(all_pixels)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        _, label = self.base_dataset[idx]
        
        # Generate Gaussian noise with matched statistics
        local_rng = np.random.RandomState(self.rng.randint(0, 2**32) + idx)
        noise = local_rng.normal(self.mean, self.std, self.img_shape)
        noise_tensor = torch.from_numpy(noise).float()
        
        return noise_tensor, label


def get_cifar10_dataset(
    root='./data',
    train=True,
    transform=None,
    download=True,
    randomization=None,
    randomization_seed=None,
    corruption_prob=0.0,
):
    """
    Get CIFAR10 dataset with optional randomization.
    
    Args:
        root: Root directory for data
        train: If True, load training set
        transform: Transformations to apply
        download: If True, download dataset if not present
        randomization: Type of randomization ('random_labels', 'partial_corrupt',
                      'shuffled_pixels', 'random_pixels', 'gaussian_pixels', or None)
        randomization_seed: Seed for randomization (ensures reproducibility)
        corruption_prob: Probability for partial corruption (only used if randomization='partial_corrupt')
    
    Returns:
        Dataset object
    """
    # Load base CIFAR10 dataset
    base_dataset = datasets.CIFAR10(
        root=root,
        train=train,
        transform=transform,
        download=download,
    )
    
    # Apply randomization wrapper if specified
    if randomization is None:
        return base_dataset
    elif randomization == 'random_labels':
        return RandomLabelDataset(base_dataset, num_classes=10, seed=randomization_seed)
    elif randomization == 'partial_corrupt':
        return PartiallyCorruptedDataset(
            base_dataset, 
            corruption_prob=corruption_prob,
            num_classes=10,
            seed=randomization_seed
        )
    elif randomization == 'shuffled_pixels':
        return ShuffledPixelsDataset(base_dataset, seed=randomization_seed)
    elif randomization == 'random_pixels':
        return RandomPixelsDataset(base_dataset, seed=randomization_seed)
    elif randomization == 'gaussian_pixels':
        return GaussianPixelsDataset(base_dataset, seed=randomization_seed)
    else:
        raise ValueError(f"Unknown randomization type: {randomization}")


def get_mnist_dataset(
    root='./data',
    train=True,
    transform=None,
    download=True,
    randomization=None,
    randomization_seed=None,
    corruption_prob=0.0,
):
    """
    Get MNIST dataset with optional randomization.
    
    Args:
        root: Root directory for data
        train: If True, load training set
        transform: Transformations to apply
        download: If True, download dataset if not present
        randomization: Type of randomization (same options as CIFAR10)
        randomization_seed: Seed for randomization
        corruption_prob: Probability for partial corruption
    
    Returns:
        Dataset object
    """
    # Load base MNIST dataset
    base_dataset = datasets.MNIST(
        root=root,
        train=train,
        transform=transform,
        download=download,
    )
    
    # Apply randomization wrapper if specified
    if randomization is None:
        return base_dataset
    elif randomization == 'random_labels':
        return RandomLabelDataset(base_dataset, num_classes=10, seed=randomization_seed)
    elif randomization == 'partial_corrupt':
        return PartiallyCorruptedDataset(
            base_dataset,
            corruption_prob=corruption_prob,
            num_classes=10,
            seed=randomization_seed
        )
    elif randomization == 'shuffled_pixels':
        return ShuffledPixelsDataset(base_dataset, seed=randomization_seed)
    elif randomization == 'random_pixels':
        return RandomPixelsDataset(base_dataset, seed=randomization_seed)
    elif randomization == 'gaussian_pixels':
        return GaussianPixelsDataset(base_dataset, seed=randomization_seed)
    else:
        raise ValueError(f"Unknown randomization type: {randomization}")
