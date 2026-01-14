"""
Shared utility modules and functions for Zhang et al. 2017 models.
Based on paper 1611.03530v2 (arXiv).
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


class ConvModule(nn.Module):
    """
    Conv Module from Zhang et al. 2017:
    Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=True):
        super(ConvModule, self).__init__()
        # When using BN, bias in conv is redundant (BN has its own bias)
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class InceptionModule2Branch(nn.Module):
    """
    2-Branch Inception Module from Zhang et al. 2017 (Figure 3):
    - Path A: 1×1 conv with ch1 filters (stride 1)
    - Path B: 3×3 conv with ch3 filters (stride 1)
    - Output: Concatenate channels (ch1 + ch3)
    """
    def __init__(self, in_channels, ch1, ch3, use_bn=True):
        """
        Args:
            in_channels: Number of input channels
            ch1: Number of filters in 1×1 conv path
            ch3: Number of filters in 3×3 conv path
            use_bn: Whether to use batch normalization
        """
        super(InceptionModule2Branch, self).__init__()
        
        # Path A: 1×1 conv
        self.path_a = ConvModule(in_channels, ch1, kernel_size=1, use_bn=use_bn)
        
        # Path B: 3×3 conv
        self.path_b = ConvModule(in_channels, ch3, kernel_size=3, padding=1, use_bn=use_bn)
    
    def forward(self, x):
        return torch.cat([self.path_a(x), self.path_b(x)], dim=1)


class DownsampleModule(nn.Module):
    """
    Downsample Module from Zhang et al. 2017 (Figure 3):
    - Path A: 3×3 conv with ch3 filters, stride 2
    - Path B: max-pool with 3×3 kernel, stride 2
    - Output: Concatenate channels (ch3 + in_channels)
    """
    def __init__(self, in_channels, ch3, use_bn=True):
        """
        Args:
            in_channels: Number of input channels
            ch3: Number of filters in 3×3 conv path
            use_bn: Whether to use batch normalization
        """
        super(DownsampleModule, self).__init__()
        
        # Path A: 3×3 conv, stride 2
        self.path_a = ConvModule(in_channels, ch3, kernel_size=3, stride=2, padding=1, use_bn=use_bn)
        
        # Path B: 3×3 max pool, stride 2
        self.path_b = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return torch.cat([self.path_a(x), self.path_b(x)], dim=1)


# Legacy 4-branch Inception module for backward compatibility
class InceptionModule(nn.Module):
    """
    Inception module with four parallel branches (GoogleNet-style):
    - 1x1 convolution
    - 1x1 -> 3x3 convolution
    - 1x1 -> 5x5 convolution
    - 3x3 max pool -> 1x1 convolution
    
    All branches are concatenated along the channel dimension.
    
    NOTE: This is NOT the module used in Zhang et al. 2017. 
    Use InceptionModule2Branch for that paper's architecture.
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


def paper_param_count(model: nn.Module) -> int:
    """
    Matches Table 1 convention from Zhang et al. 2017:
    count Conv/FC params but EXCLUDE BatchNorm params.

    Args:
        model: PyTorch model

    Returns:
        int: Number of parameters excluding BatchNorm
    """
    total = 0
    for name, p in model.named_parameters():
        if ".bn." not in name and "bn." not in name:
            total += p.numel()
    return total


def total_param_count(model: nn.Module) -> int:
    """
    Count total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())
