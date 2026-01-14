"""
Data transformations for CIFAR-10 (Zhang et al. 2017 experiments).

Implements the paper's preprocessing pipeline:
- CIFAR-10: ALWAYS cropped from 32×32 to 28×28 (center crop or random crop)
- Per-image whitening (subtract mean, divide by adjusted stddev)
- Optional augmentations: random crop (vs center crop), flip, rotation
"""
import torchvision.transforms as transforms
import torch

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


def get_cifar10_transforms(random_crop=False, augment_flip_rotate=False, center_crop_size=28):
    t = []

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

    return transforms.Compose(t)
