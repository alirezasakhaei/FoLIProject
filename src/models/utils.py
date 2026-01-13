"""
Shared utility modules and functions for Zhang et al. 2017 models.
"""
import torch
import torch.nn as nn


class Flatten(nn.Module):
    """Flattens input tensor to (batch_size, -1)."""
    def forward(self, x):
        return x.view(x.size(0), -1)


def get_flattened_size(model_fe, input_shape):
    """
    Compute the flattened feature size after passing through a feature extractor.
    
    Args:
        model_fe: Feature extractor module
        input_shape: Tuple of (C, H, W) for input dimensions
        
    Returns:
        int: Flattened feature size
    """
    dummy_input = torch.zeros(1, *input_shape)
    with torch.no_grad():
        output = model_fe(dummy_input)
    return output.view(1, -1).size(1)


class InceptionModule(nn.Module):
    """
    Inception module with four parallel branches:
    - 1x1 convolution
    - 1x1 -> 3x3 convolution
    - 1x1 -> 5x5 convolution
    - 3x3 max pool -> 1x1 convolution
    
    All branches are concatenated along the channel dimension.
    """
    def __init__(self, in_channels, n1x1, n3x3reduce, n3x3, n5x5reduce, n5x5, pool_proj, use_bn=True):
        """
        Args:
            in_channels: Number of input channels
            n1x1: Number of 1x1 filters in the 1x1 branch
            n3x3reduce: Number of 1x1 filters before 3x3 conv
            n3x3: Number of 3x3 filters
            n5x5reduce: Number of 1x1 filters before 5x5 conv
            n5x5: Number of 5x5 filters
            pool_proj: Number of 1x1 filters after pooling
            use_bn: Whether to use batch normalization
        """
        super(InceptionModule, self).__init__()
        self.use_bn = use_bn
        
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1) if use_bn else nn.Identity(),
            nn.ReLU(True)
        )

        # 1x1 -> 3x3 branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3reduce) if use_bn else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(n3x3reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3) if use_bn else nn.Identity(),
            nn.ReLU(True)
        )

        # 1x1 -> 5x5 branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5reduce) if use_bn else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(n5x5reduce, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5) if use_bn else nn.Identity(),
            nn.ReLU(True)
        )

        # 3x3 pool -> 1x1 branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj) if use_bn else nn.Identity(),
            nn.ReLU(True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)
