"""
Data transformations for CIFAR-10 (Zhang et al. 2017 experiments).

Implements the paper's preprocessing pipeline:
- CIFAR-10: ALWAYS cropped from 32×32 to 28×28 (center crop or random crop)
- Per-image whitening (subtract mean, divide by adjusted stddev)
- Optional augmentations: random crop (vs center crop), flip, rotation
- Randomizations:
  - Shuffled pixels: Same permutation applied to all train+test (BEFORE whitening)
  - Random pixels: Different permutation per training image, test unchanged (BEFORE whitening)
  - Gaussian noise: Gaussian noise with mean 0, std 1 (AFTER whitening)
"""
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image

class ShuffledPixels:
    """Apply a fixed permutation to all pixels in the image."""
    def __init__(self, permutation):
        """
        Args:
            permutation: A fixed permutation index array for all images
        """
        self.permutation = permutation

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert PIL image to numpy array
        img_array = np.array(img)
        h, w, c = img_array.shape

        # Flatten and permute
        flat = img_array.reshape(-1, c)
        permuted = flat[self.permutation]

        # Reshape back
        result = permuted.reshape(h, w, c)
        return Image.fromarray(result.astype(np.uint8))


class RandomPixels:
    """Apply a different random permutation to each image."""
    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert PIL image to numpy array
        img_array = np.array(img)
        h, w, c = img_array.shape

        # Flatten and permute with random permutation
        flat = img_array.reshape(-1, c)
        permutation = np.random.permutation(len(flat))
        permuted = flat[permutation]

        # Reshape back
        result = permuted.reshape(h, w, c)
        return Image.fromarray(result.astype(np.uint8))


class GaussianPixels:
    """Replace image with Gaussian noise matching dataset statistics.

    NOTE: This operates on tensors AFTER whitening, not on PIL images.
    The noise is generated with mean 0 and std 1 to match the whitened distribution.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (C, H, W) after whitening (mean ~0, std ~1)

        Returns:
            Tensor of same shape with Gaussian noise (mean 0, std 1)
        """
        # Generate Gaussian noise with mean 0, std 1 (matching whitened images)
        noise = torch.randn_like(x)
        return noise


class PerImageWhitening:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W), float32 in [0,1]
        mean = x.mean()
        std = x.std(unbiased=False)

        n = x.numel()
        min_std = (1.0 / (n ** 0.5))
        min_std = torch.tensor(min_std, device=x.device, dtype=x.dtype)

        adjusted_std = torch.maximum(std, min_std)
        return (x - mean) / adjusted_std


def get_cifar10_transforms(random_crop=False, augment_flip_rotate=False, center_crop_size=28, pixel_randomization=None):
    """
    Get CIFAR-10 transforms with optional pixel randomization.

    Args:
        random_crop: Use random crop instead of center crop
        augment_flip_rotate: Apply random flip and rotation
        center_crop_size: Size of the crop (default 28 for CIFAR-10)
        pixel_randomization: Optional pixel randomization transform (ShuffledPixels, RandomPixels, or GaussianPixels)
    """
    t = []

    # Apply ShuffledPixels or RandomPixels BEFORE whitening (on PIL images)
    # GaussianPixels is applied AFTER whitening (on tensors)
    if pixel_randomization is not None and not isinstance(pixel_randomization, GaussianPixels):
        t.append(pixel_randomization)

    # Optional aug (do on PIL)
    if augment_flip_rotate:
        t.append(transforms.RandomHorizontalFlip(p=0.5))
        t.append(transforms.RandomRotation(degrees=25))

    # Crop to 28x28 (paper always uses 28x28 inputs)
    if random_crop:
        t.append(transforms.RandomCrop(center_crop_size))
    else:
        t.append(transforms.CenterCrop(center_crop_size))

    # Scale to [0,1], then whiten per image
    t.append(transforms.ToTensor())
    t.append(PerImageWhitening())

    # Apply GaussianPixels AFTER whitening (on tensors with mean ~0, std ~1)
    if pixel_randomization is not None and isinstance(pixel_randomization, GaussianPixels):
        t.append(pixel_randomization)

    return transforms.Compose(t)


def get_imagenet_transforms(random_crop=False, augment_flip_rotate=False, crop_size=224, pixel_randomization=None):
    """
    Get ImageNet transforms with optional pixel randomization.

    Args:
        random_crop: Use random crop instead of center crop
        augment_flip_rotate: Apply random flip and rotation
        crop_size: Size of the crop (default 224 for ImageNet)
        pixel_randomization: Optional pixel randomization transform (ShuffledPixels, RandomPixels, or GaussianPixels)
    """
    t = []

    # Apply ShuffledPixels or RandomPixels BEFORE whitening (on PIL images)
    # GaussianPixels is applied AFTER whitening (on tensors)
    if pixel_randomization is not None and not isinstance(pixel_randomization, GaussianPixels):
        t.append(pixel_randomization)

    # Resize to 256x256 first (standard ImageNet preprocessing)
    t.append(transforms.Resize(256))

    # Optional aug (do on PIL)
    if augment_flip_rotate:
        t.append(transforms.RandomHorizontalFlip(p=0.5))
        t.append(transforms.RandomRotation(degrees=25))

    # Crop to target size (224x224 by default)
    if random_crop:
        t.append(transforms.RandomCrop(crop_size))
    else:
        t.append(transforms.CenterCrop(crop_size))

    # Scale to [0,1], then whiten per image
    t.append(transforms.ToTensor())
    t.append(PerImageWhitening())

    # Apply GaussianPixels AFTER whitening (on tensors with mean ~0, std ~1)
    if pixel_randomization is not None and isinstance(pixel_randomization, GaussianPixels):
        t.append(pixel_randomization)

    return transforms.Compose(t)
