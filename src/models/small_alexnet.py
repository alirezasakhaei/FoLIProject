"""
Small AlexNet from Zhang et al. 2017.
Adapted for CIFAR-10 with smaller spatial dimensions.
"""
import torch.nn as nn
from .utils import get_flattened_size


class SmallAlexNet(nn.Module):
    """
    Small AlexNet adapted for CIFAR-10.
    
    Architecture:
    - Two blocks of: Conv 5x5 -> MaxPool 3x3 -> Local Response Norm
    - FC(384) -> FC(192) -> Linear(num_classes)
    """
    def __init__(self, num_classes=10, input_shape=(3, 32, 32)):
        """
        Args:
            num_classes: Number of output classes
            input_shape: Tuple of (C, H, W) for input dimensions
        """
        super(SmallAlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_shape[0], 96, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            
            # Second conv block
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
        )
        
        # Compute flattened size dynamically based on input shape
        flat_size = get_flattened_size(self.features, input_shape)
        
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def small_alexnet(num_classes=10, input_shape=(3, 32, 32)):
    """Small AlexNet for CIFAR-10."""
    return SmallAlexNet(num_classes, input_shape)
