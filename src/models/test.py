"""
Test script for all models from Zhang et al. 2017.
Loads each model and displays parameter counts.

Note: All CIFAR-10 models use 28×28 inputs (cropped from 32×32).
"""
import torch
from .utils import paper_param_count, total_param_count
from . import (
    inception,
    inception_no_bn,
    small_alexnet,
    mlp_1x512,
    mlp_3x512,
)


def test_model(name, model_fn, input_shape=(3, 32, 32)):
    """Test a single model and print parameter counts."""
    print(f"\n{'='*70}")
    print(f"Model: {name}")
    print(f"{'='*70}")

    # Create model
    model = model_fn(num_classes=10, input_shape=input_shape)

    # Count parameters
    paper_params = paper_param_count(model)
    total_params = total_param_count(model)

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, *input_shape)
    with torch.no_grad():
        y = model(x)

    print(f"Input shape:             {tuple(x.shape)}")
    print(f"Output shape:            {tuple(y.shape)}")
    print(f"Paper params (no BN):    {paper_params:,}")
    print(f"Total params (with BN):  {total_params:,}")

    # Calculate BatchNorm params
    bn_params = total_params - paper_params
    if bn_params > 0:
        print(f"BatchNorm params:        {bn_params:,}")

    return {
        'name': name,
        'paper_params': paper_params,
        'total_params': total_params,
        'bn_params': bn_params,
        'output_shape': tuple(y.shape),
    }


def main():
    """Test all models."""
    print("\n" + "="*70)
    print("TESTING ALL MODELS FROM ZHANG ET AL. 2017")
    print("="*70)

    # Define all models to test with their input shapes
    # ALL CIFAR-10 models use 28×28 inputs (always cropped from 32×32)
    models = [
        ('inception', inception, (3, 28, 28)),
        ('inception_no_bn', inception_no_bn, (3, 28, 28)),
        ('small_alexnet', small_alexnet, (3, 28, 28)),
        ('mlp_1x512', mlp_1x512, (3, 28, 28)),
        ('mlp_3x512', mlp_3x512, (3, 28, 28)),
    ]

    # Test each model
    results = []
    for name, model_fn, input_shape in models:
        result = test_model(name, model_fn, input_shape)
        results.append(result)

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Paper Params':<15} {'Total Params':<15} {'BN Params':<12}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*12}")

    for r in results:
        print(f"{r['name']:<20} {r['paper_params']:>15,} {r['total_params']:>15,} {r['bn_params']:>12,}")

if __name__ == "__main__":
    main()
