"""
Utility functions for dataset.
"""

def num_class_counts(dataset: str) -> int:
    """
    Get the number of class counts for a dataset.
    """
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'imagenet':
        return 1000
    else:
        raise ValueError(f"Dataset {dataset} not supported")