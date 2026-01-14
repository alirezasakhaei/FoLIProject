"""
Data transformations for CIFAR-10 (Zhang et al. 2017 experiments).

Implements the paper's preprocessing pipeline:
- CIFAR-10: ALWAYS cropped from 32×32 to 28×28 (center crop or random crop)
- Per-image whitening (subtract mean, divide by adjusted stddev)
- Optional augmentations: random crop (vs center crop), flip, rotation
"""
import torchvision.transforms as transforms


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


def get_cifar10_transforms(
    random_crop=False,
    augment_flip_rotate=False,
    center_crop_size=28,
):
    """
    Get CIFAR-10 transforms following Zhang et al. 2017.

    IMPORTANT: CIFAR-10 images are ALWAYS cropped from 32×32 to 28×28 in this paper.
    The random_crop flag controls HOW the crop is performed:
    - random_crop=False: Center crop to 28×28 (baseline, test set always uses this)
    - random_crop=True: Random 28×28 crop from 32×32 (data augmentation, Table 1)

    Base preprocessing (always applied):
    - Convert to tensor (gives [0, 1] range)
    - Crop to 28×28 (center crop OR random crop based on flag)
    - Per-image whitening (subtract mean, divide by adjusted stddev)

    Args:
        random_crop: If True, use random 28×28 crop instead of center crop (Table 1)
        augment_flip_rotate: If True, apply random flip + rotation ≤25° (Appendix E)
        center_crop_size: Crop size (default 28, matching paper)

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
