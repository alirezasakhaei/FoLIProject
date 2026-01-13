"""
Small Inception network from Zhang et al. 2017.
Supports both with and without batch normalization variants.
"""
import torch.nn as nn
from .utils import InceptionModule


class SmallInception(nn.Module):
    """
    Small Inception network adapted for CIFAR-10.
    
    Architecture:
    - Initial 3x3 conv with 96 filters
    - Stage 1: 2 Inception modules + max pool downsample
    - Stage 2: 2 Inception modules
    - Global average pooling
    - Linear classifier
    """
    def __init__(self, num_classes=10, input_shape=(3, 32, 32), use_bn=True):
        """
        Args:
            num_classes: Number of output classes
            input_shape: Tuple of (C, H, W) for input dimensions
            use_bn: Whether to use batch normalization
        """
        super(SmallInception, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96) if use_bn else nn.Identity(),
            nn.ReLU(True),
            
            # Stage 1
            InceptionModule(96, 32, 32, 32, 8, 8, 32, use_bn=use_bn),  # Out: 32+32+8+32 = 104
            InceptionModule(104, 32, 32, 48, 8, 8, 32, use_bn=use_bn),  # Out: 32+48+8+32 = 120
            nn.MaxPool2d(3, stride=2, padding=1),  # Downsample
            
            # Stage 2
            InceptionModule(120, 112, 32, 48, 8, 32, 48, use_bn=use_bn),  # Out: 112+48+32+48 = 240
            InceptionModule(240, 160, 112, 224, 24, 64, 64, use_bn=use_bn),  # Out: 160+224+64+64 = 512
        )
        
        # Global average pooling to handle variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def small_inception(num_classes=10, input_shape=(3, 32, 32)):
    """Small Inception with batch normalization."""
    return SmallInception(num_classes, input_shape, use_bn=True)


def small_inception_no_bn(num_classes=10, input_shape=(3, 32, 32)):
    """Small Inception without batch normalization."""
    return SmallInception(num_classes, input_shape, use_bn=False)
