"""
Inception network from Zhang et al. 2017 (without Batch Normalization).
Architecture from paper 1611.03530v2, Table 1.
"""
import torch.nn as nn
from .utils import InceptionModule


class InceptionNoBN(nn.Module):
    """
    Inception network adapted for CIFAR-10 (without Batch Normalization).
    
    Architecture:
    - Initial 3x3 conv with 96 filters (no BN)
    - Stage 1: 2 Inception modules + max pool downsample
    - Stage 2: 2 Inception modules
    - Global average pooling
    - Linear classifier
    
    Parameters: 1,649,402
    """
    def __init__(self, num_classes=10, input_shape=(3, 32, 32)):
        """
        Args:
            num_classes: Number of output classes
            input_shape: Tuple of (C, H, W) for input dimensions
        """
        super(InceptionNoBN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 96, kernel_size=3, padding=1),
            nn.ReLU(True),
            
            # Stage 1
            InceptionModule(96, 32, 32, 32, 8, 8, 32, use_bn=False),  # Out: 32+32+8+32 = 104
            InceptionModule(104, 32, 32, 48, 8, 8, 32, use_bn=False),  # Out: 32+48+8+32 = 120
            nn.MaxPool2d(3, stride=2, padding=1),  # Downsample
            
            # Stage 2
            InceptionModule(120, 112, 32, 48, 8, 32, 48, use_bn=False),  # Out: 112+48+32+48 = 240
            InceptionModule(240, 160, 112, 224, 24, 64, 64, use_bn=False),  # Out: 160+224+64+64 = 512
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


def inception_no_bn(num_classes=10, input_shape=(3, 32, 32)):
    """Inception without batch normalization (1,649,402 params)."""
    return InceptionNoBN(num_classes, input_shape)
