"""
Test script to verify data pipeline and configuration system.

This tests:
1. Transforms (per-image whitening, center crop, augmentation)
2. Dataset wrappers (random labels, pixel randomization)
3. Configuration system
"""
import torch
import numpy as np
from src.data import (
    get_cifar10_transforms,
    get_mnist_transforms,
    get_cifar10_dataset,
    get_mnist_dataset,
)
from src.config import (
    ExperimentConfig,
    get_baseline_config,
    get_random_labels_config,
    get_optimizer,
)
from src.models import inception


def test_transforms():
    """Test transform pipeline."""
    print("="*60)
    print("Testing Transforms")
    print("="*60)
    
    # Test CIFAR10 transforms
    print("\n1. CIFAR10 base transforms (center crop + whitening):")
    transform = get_cifar10_transforms(
        random_crop=False,
        augment_flip_rotate=False,
        center_crop_size=28,
    )
    print(f"   Transform: {transform}")
    
    # Create a dummy image
    from PIL import Image
    dummy_img = Image.new('RGB', (32, 32), color=(128, 128, 128))
    transformed = transform(dummy_img)
    print(f"   Output shape: {transformed.shape}")
    print(f"   Output dtype: {transformed.dtype}")
    print(f"   Output mean: {transformed.mean():.4f}")
    print(f"   Output std: {transformed.std():.4f}")
    assert transformed.shape == (3, 28, 28), "Expected (3, 28, 28)"
    print("   ✓ Base transforms working")
    
    # Test with augmentation
    print("\n2. CIFAR10 with augmentation (flip + rotate):")
    transform_aug = get_cifar10_transforms(
        random_crop=False,
        augment_flip_rotate=True,
        center_crop_size=28,
    )
    transformed_aug = transform_aug(dummy_img)
    print(f"   Output shape: {transformed_aug.shape}")
    assert transformed_aug.shape == (3, 28, 28), "Expected (3, 28, 28)"
    print("   ✓ Augmentation transforms working")
    
    # Test MNIST transforms
    print("\n3. MNIST transforms:")
    mnist_transform = get_mnist_transforms(
        random_crop=False,
        augment_flip_rotate=False,
    )
    dummy_mnist = Image.new('L', (28, 28), color=128)
    transformed_mnist = mnist_transform(dummy_mnist)
    print(f"   Output shape: {transformed_mnist.shape}")
    assert transformed_mnist.shape == (1, 28, 28), "Expected (1, 28, 28)"
    print("   ✓ MNIST transforms working")


def test_datasets():
    """Test dataset wrappers."""
    print("\n" + "="*60)
    print("Testing Dataset Wrappers")
    print("="*60)
    
    # Test base CIFAR10
    print("\n1. Base CIFAR10 dataset:")
    transform = get_cifar10_transforms()
    dataset = get_cifar10_dataset(
        root='./data',
        train=True,
        transform=transform,
        download=True,
        randomization=None,
    )
    print(f"   Dataset size: {len(dataset)}")
    img, label = dataset[0]
    print(f"   Sample shape: {img.shape}")
    print(f"   Sample label: {label}")
    print("   ✓ Base dataset working")
    
    # Test random labels
    print("\n2. Random labels dataset:")
    random_dataset = get_cifar10_dataset(
        root='./data',
        train=True,
        transform=transform,
        download=True,
        randomization='random_labels',
        randomization_seed=42,
    )
    img1, label1 = random_dataset[0]
    img2, label2 = random_dataset[0]  # Same index should give same random label
    print(f"   First access label: {label1}")
    print(f"   Second access label: {label2}")
    assert label1 == label2, "Random labels should be consistent!"
    print("   ✓ Random labels are consistent across accesses")
    
    # Test partially corrupted
    print("\n3. Partially corrupted dataset (50% corruption):")
    corrupt_dataset = get_cifar10_dataset(
        root='./data',
        train=True,
        transform=transform,
        download=True,
        randomization='partial_corrupt',
        corruption_prob=0.5,
        randomization_seed=42,
    )
    print(f"   Dataset size: {len(corrupt_dataset)}")
    print("   ✓ Partial corruption working")
    
    # Test shuffled pixels
    print("\n4. Shuffled pixels dataset:")
    shuffled_dataset = get_cifar10_dataset(
        root='./data',
        train=True,
        transform=transform,
        download=True,
        randomization='shuffled_pixels',
        randomization_seed=42,
    )
    img_shuffled, _ = shuffled_dataset[0]
    print(f"   Shuffled image shape: {img_shuffled.shape}")
    print("   ✓ Shuffled pixels working")
    
    # Test random pixels
    print("\n5. Random pixels dataset:")
    random_pix_dataset = get_cifar10_dataset(
        root='./data',
        train=True,
        transform=transform,
        download=True,
        randomization='random_pixels',
        randomization_seed=42,
    )
    img_random, _ = random_pix_dataset[0]
    print(f"   Random pixels image shape: {img_random.shape}")
    print("   ✓ Random pixels working")
    
    # Test Gaussian pixels
    print("\n6. Gaussian pixels dataset:")
    gaussian_dataset = get_cifar10_dataset(
        root='./data',
        train=True,
        transform=transform,
        download=True,
        randomization='gaussian_pixels',
        randomization_seed=42,
    )
    img_gaussian, _ = gaussian_dataset[0]
    print(f"   Gaussian image shape: {img_gaussian.shape}")
    print("   ✓ Gaussian pixels working")


def test_config():
    """Test configuration system."""
    print("\n" + "="*60)
    print("Testing Configuration System")
    print("="*60)
    
    # Test baseline config
    print("\n1. Baseline config:")
    config = get_baseline_config()
    print(f"   Model: {config.model_name}")
    print(f"   Randomization: {config.randomization}")
    print(f"   Weight decay: {config.weight_decay}")
    print(f"   Explicit reg off: {config.explicit_reg_off}")
    assert config.explicit_reg_off == True, "Baseline should have explicit_reg_off=True"
    print("   ✓ Baseline config correct")
    
    # Test random labels config
    print("\n2. Random labels config:")
    config_random = get_random_labels_config()
    print(f"   Randomization: {config_random.randomization}")
    print(f"   Explicit reg off: {config_random.explicit_reg_off}")
    assert config_random.randomization == 'random_labels'
    assert config_random.explicit_reg_off == True
    print("   ✓ Random labels config correct")
    
    # Test custom config
    print("\n3. Custom config with overrides:")
    config_custom = ExperimentConfig(
        model_name='small_alexnet',
        weight_decay=0.0005,
        random_crop=True,
    )
    print(f"   Model: {config_custom.model_name}")
    print(f"   Weight decay: {config_custom.weight_decay}")
    print(f"   Random crop: {config_custom.random_crop}")
    print(f"   Explicit reg off: {config_custom.explicit_reg_off}")
    assert config_custom.explicit_reg_off == False, "Should have explicit_reg_off=False"
    print("   ✓ Custom config correct")
    
    # Test optimizer creation
    print("\n4. Optimizer creation:")
    model = inception()
    optimizer = get_optimizer(model, config_custom)
    print(f"   Optimizer type: {type(optimizer).__name__}")
    print(f"   Weight decay: {optimizer.param_groups[0]['weight_decay']}")
    assert optimizer.param_groups[0]['weight_decay'] == 0.0005
    print("   ✓ Optimizer creation correct")


def test_integration():
    """Test full integration."""
    print("\n" + "="*60)
    print("Testing Full Integration")
    print("="*60)
    
    print("\n1. Creating model + data pipeline:")
    config = ExperimentConfig(
        model_name='inception',
        dataset='cifar10',
        randomization='random_labels',
        batch_size=4,
        input_shape=(3, 28, 28),  # After center crop
    )
    
    # Get model
    model = inception(num_classes=10, input_shape=(3, 28, 28))
    print(f"   Model: {config.model_name}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get dataset
    transform = get_cifar10_transforms()
    dataset = get_cifar10_dataset(
        root='./data',
        train=True,
        transform=transform,
        download=True,
        randomization=config.randomization,
        randomization_seed=42,
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Test forward pass
    print("\n2. Testing forward pass:")
    batch_data, batch_labels = next(iter(loader))
    print(f"   Batch data shape: {batch_data.shape}")
    print(f"   Batch labels shape: {batch_labels.shape}")
    
    output = model(batch_data)
    print(f"   Output shape: {output.shape}")
    assert output.shape == (config.batch_size, 10), "Expected (batch_size, 10)"
    print("   ✓ Forward pass successful")
    
    # Test loss computation
    print("\n3. Testing loss computation:")
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, batch_labels)
    print(f"   Loss: {loss.item():.4f}")
    print("   ✓ Loss computation successful")
    
    print("\n" + "="*60)
    print("All integration tests passed!")
    print("="*60)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Zhang et al. 2017 - Data Pipeline Test Suite")
    print("="*60)
    
    try:
        test_transforms()
        test_datasets()
        test_config()
        test_integration()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now run experiments with:")
        print("  python train.py --model inception --dataset cifar10")
        print("  python train.py --model inception --randomization random_labels")
        print("  python train.py --model inception --weight_decay 0.0005 --random_crop")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
