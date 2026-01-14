"""
Small Inception network from Zhang et al. 2017 (with Batch Normalization).
Architecture from paper 1611.03530v2, Figure 3 and Table 1.

Exact Figure 3 graph (CIFAR-10 center-cropped to 28x28).
Note: conv_bias=True is required to match Table 1 parameter count.
"""
import torch
import torch.nn as nn


class ConvModule(nn.Module):
    # Figure 3: Conv -> BN -> ReLU
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int, conv_bias: bool = True):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, bias=conv_bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InceptionModule(nn.Module):
    # Figure 3: parallel 1x1 and 3x3 ConvModules, concat channels
    def __init__(self, in_ch: int, ch1: int, ch3: int, conv_bias: bool = True):
        super().__init__()
        self.b1 = ConvModule(in_ch, ch1, k=1, stride=1, conv_bias=conv_bias)
        self.b3 = ConvModule(in_ch, ch3, k=3, stride=1, conv_bias=conv_bias)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x)], dim=1)


class DownsampleModule(nn.Module):
    # Figure 3: parallel (3x3 stride2 ConvModule) and (3x3 stride2 MaxPool), concat channels
    def __init__(self, in_ch: int, ch3: int, conv_bias: bool = True):
        super().__init__()
        self.conv = ConvModule(in_ch, ch3, k=3, stride=2, conv_bias=conv_bias)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.conv(x), self.pool(x)], dim=1)


class SmallInceptionCIFAR10(nn.Module):
    """
    Exact Figure 3 graph (CIFAR-10 center-cropped to 28x28).
    Note: conv_bias=True is required to match Table 1 parameter count.
    """
    def __init__(self, num_classes: int = 10, conv_bias: bool = True):
        super().__init__()

        # 28x28x3 -> 28x28x96
        self.stem = ConvModule(3, 96, k=3, stride=1, conv_bias=conv_bias)

        # Left column
        self.i1 = InceptionModule(96, 32, 32, conv_bias=conv_bias)   # -> 64
        self.i2 = InceptionModule(64, 32, 48, conv_bias=conv_bias)   # -> 80
        self.d1 = DownsampleModule(80, 80, conv_bias=conv_bias)      # -> 160, 14x14

        # Middle column (TOP -> BOTTOM exactly as in Figure 3)
        self.i3 = InceptionModule(160, 112, 48, conv_bias=conv_bias) # -> 160
        self.i4 = InceptionModule(160, 96, 64, conv_bias=conv_bias)  # -> 160
        self.i5 = InceptionModule(160, 80, 80, conv_bias=conv_bias)  # -> 160
        self.i6 = InceptionModule(160, 48, 96, conv_bias=conv_bias)  # -> 144
        self.d2 = DownsampleModule(144, 96, conv_bias=conv_bias)     # -> 240, 7x7

        # Right column
        self.i7 = InceptionModule(240, 176, 160, conv_bias=conv_bias) # -> 336
        self.i8 = InceptionModule(336, 176, 160, conv_bias=conv_bias) # -> 336

        self.pool = nn.AvgPool2d(kernel_size=7, stride=1)  # global 7x7
        self.fc = nn.Linear(336, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.i1(x); x = self.i2(x); x = self.d1(x)
        x = self.i3(x); x = self.i4(x); x = self.i5(x); x = self.i6(x); x = self.d2(x)
        x = self.i7(x); x = self.i8(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def paper_param_count(model: nn.Module) -> int:
    """
    Matches Table 1 convention:
    count Conv/FC params but EXCLUDE BatchNorm params.
    """
    total = 0
    for name, p in model.named_parameters():
        if ".bn." in name:
            continue
        total += p.numel()
    return total


def inception(num_classes=10, input_shape=(3, 32, 32)):
    """
    Small Inception with batch normalization.
    Target: 1,649,402 parameters.
    """
    return SmallInceptionCIFAR10(num_classes=num_classes, conv_bias=True)


if __name__ == "__main__":
    model = SmallInceptionCIFAR10(num_classes=10, conv_bias=True)

    x = torch.randn(2, 3, 28, 28)
    y = model(x)
    print("Output shape:", tuple(y.shape))

    print("Paper-style #params (exclude BN, include conv biases):", paper_param_count(model))