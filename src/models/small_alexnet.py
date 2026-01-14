"""
Small AlexNet from Zhang et al. 2017.
Adapted for CIFAR-10 (28×28 input after cropping).
"""
import torch
import torch.nn as nn


class SmallAlexNet(nn.Module):
    """
    Paper description:
    - Two (Conv 5x5 → MaxPool 3x3 → LocalResponseNorm) modules
    - FC(384) → FC(192) → Linear(10)
    - ReLU activations
    
    """
    def __init__(self, num_classes: int = 10, input_shape=(3, 28, 28)):
        super().__init__()

        c, h, w = input_shape
        if (h, w) != (28, 28):
            raise ValueError(f"Paper CIFAR setup uses 28x28 inputs; got {h}x{w}.")

        # IMPORTANT: channel sizes chosen to match Table 1: 1,387,786 params. :contentReference[oaicite:3]{index=3}
        conv1_out = 64
        conv2_out = 64

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(c, conv1_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),

            # Block 2
            nn.Conv2d(conv1_out, conv2_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
        )

        # With 28x28 input and the above padding/pooling: 28 -> 14 -> 7, channels=64
        flat_size = conv2_out * 7 * 7  # 3136

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 384, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def small_alexnet(num_classes: int = 10, input_shape=(3, 28, 28)):
    return SmallAlexNet(num_classes=num_classes, input_shape=input_shape)
