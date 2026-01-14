"""
Model definitions for Zhang et al. 2017 experiments.
"""
from .inception import inception
from .inception_no_bn import inception_no_bn
from .small_alexnet import small_alexnet
from .mlp import mlp_1x512, mlp_3x512

__all__ = [
    'inception',
    'inception_no_bn',
    'small_alexnet',
    'mlp_1x512',
    'mlp_3x512',
]
