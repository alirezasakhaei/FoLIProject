"""
Zhang et al. 2017 models for studying generalization in deep learning.
"""
from .small_inception import small_inception, small_inception_no_bn
from .small_alexnet import small_alexnet
from .mlp import mlp_1x512, mlp_3x512

__all__ = [
    'small_inception',
    'small_inception_no_bn',
    'small_alexnet',
    'mlp_1x512',
    'mlp_3x512',
]
