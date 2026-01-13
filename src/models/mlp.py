"""
Multi-Layer Perceptron (MLP) models from Zhang et al. 2017.
Simple fully-connected networks for baseline comparisons.
"""
import torch.nn as nn
from functools import reduce
from operator import mul


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable depth and width.
    
    Architecture:
    - Flatten input
    - n_hidden_layers of Linear -> ReLU
    - Final linear layer for classification
    """
    def __init__(self, n_hidden_layers, hidden_size, num_classes=10, input_shape=(3, 32, 32)):
        """
        Args:
            n_hidden_layers: Number of hidden layers
            hidden_size: Number of units in each hidden layer
            num_classes: Number of output classes
            input_shape: Tuple of (C, H, W) for input dimensions
        """
        super(MLP, self).__init__()
        self.input_size = reduce(mul, input_shape)
        
        layers = []
        in_dim = self.input_size
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_size
        
        layers.append(nn.Linear(in_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


def mlp_1x512(num_classes=10, input_shape=(3, 32, 32)):
    """MLP with 1 hidden layer of 512 units."""
    return MLP(1, 512, num_classes, input_shape)


def mlp_3x512(num_classes=10, input_shape=(3, 32, 32)):
    """MLP with 3 hidden layers of 512 units each."""
    return MLP(3, 512, num_classes, input_shape)
