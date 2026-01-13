# Zhang et al. 2017: Understanding Deep Learning Requires Rethinking Generalization

Implementation of models and experiments from the paper ["Understanding deep learning requires rethinking generalization"](https://arxiv.org/abs/1611.03530) by Zhang et al. (ICLR 2017, Best Paper Award).

## Overview

This repository implements the paper-faithful experiments for CIFAR-10 and MNIST, including:
- 5 neural network architectures
- Complete data preprocessing pipeline
- All randomization operators (random labels, pixel shuffling, etc.)
- Regularization toggles (weight decay, data augmentation, BatchNorm)

## Models

All models support both 32×32 (CIFAR-10) and 224×224 (ImageNet-scale) inputs:

1. **Small Inception** - Inception-style network with BatchNorm
2. **Small Inception (No BN)** - Same architecture without BatchNorm
3. **Small AlexNet** - AlexNet adapted for CIFAR-10 with Local Response Normalization
4. **MLP 1×512** - Single hidden layer with 512 units
5. **MLP 3×512** - Three hidden layers with 512 units each

## Installation

```bash
# Install dependencies
uv pip install -r requirements.txt
```

## Quick Start

### Test the models
```bash
python test_zhang_models.py
```

### Test the data pipeline
```bash
python test_data_pipeline.py
```

### Run a basic experiment
```bash
# Train Small Inception on CIFAR-10 (baseline)
python train.py --model small_inception --dataset cifar10

# Train with random labels (Table 1 experiment)
python train.py --model small_inception --randomization random_labels

# Train with weight decay and random crop (regularization)
python train.py --model small_inception --weight_decay 0.0005 --random_crop
```

## Paper-Critical Features

### 1. Preprocessing (Section 4, Appendix A)

**CIFAR-10:**
- Divide by 255 to [0, 1]
- Center crop to 28×28
- Per-image whitening (subtract mean, divide by adjusted std)

**MNIST:**
- Divide by 255 to [0, 1]
- Per-image whitening

### 2. Regularization Toggles

#### Weight Decay (Table 1, Table 4)
```bash
python train.py --weight_decay 0.0005
```
- Implemented in optimizer (SGD with L2 regularization)
- Default coefficients per model (configurable)

#### Data Augmentation

**Random Crop (Table 1):**
```bash
python train.py --random_crop
```
- Standard CIFAR-style random crop to 28×28

**Flip + Rotation (Appendix E, Table 4):**
```bash
python train.py --augment_flip_rotate
```
- Random horizontal flip (p=0.5)
- Random rotation up to 25°

#### BatchNorm Toggle
```bash
# With BatchNorm (default)
python train.py --model small_inception

# Without BatchNorm
python train.py --model small_inception_no_bn
```

### 3. Randomization Operators (Table 1, Appendix E)

All randomization patterns are **fixed at initialization** and **consistent across epochs** (as per paper methodology).

#### Random Labels
```bash
python train.py --randomization random_labels
```
- All labels replaced with random ones
- Labels remain consistent across epochs

#### Partially Corrupted Labels
```bash
python train.py --randomization partial_corrupt --corruption_prob 0.5
```
- Each label has probability `p` of being random
- Corruption pattern fixed at initialization

#### Shuffled Pixels
```bash
python train.py --randomization shuffled_pixels
```
- Same pixel permutation applied to all images
- Preserves pixel values, destroys spatial structure

#### Random Pixels
```bash
python train.py --randomization random_pixels
```
- Different random permutation per image
- Destroys all spatial structure

#### Gaussian Pixels
```bash
python train.py --randomization gaussian_pixels
```
- Replace images with Gaussian noise
- Matched mean and variance to original dataset

### 4. "No Explicit Regularization" Setting

For random label experiments (Table 1, Appendix E):
```bash
python train.py --randomization random_labels
# Automatically sets: weight_decay=0, augmentation=False
```

The config system tracks this with `config.explicit_reg_off` property.

## Reproducing Paper Experiments

### Table 1: CIFAR-10 with Different Regularization

```bash
# Inception (baseline)
python train.py --model small_inception --dataset cifar10

# Inception with random labels
python train.py --model small_inception --randomization random_labels

# Inception with shuffled pixels
python train.py --model small_inception --randomization shuffled_pixels

# Inception with random pixels
python train.py --model small_inception --randomization random_pixels

# Inception with Gaussian noise
python train.py --model small_inception --randomization gaussian_pixels
```

### Table 4 (Appendix E): Augmentation on Random Labels

```bash
# Random labels without augmentation
python train.py --model small_inception --randomization random_labels

# Random labels with augmentation
python train.py --model small_inception --randomization random_labels --augment_flip_rotate

# Random labels with weight decay
python train.py --model small_inception --randomization random_labels --weight_decay 0.0005
```

### BatchNorm Comparison

```bash
# Small Inception with BatchNorm
python train.py --model small_inception --dataset cifar10

# Small Inception without BatchNorm
python train.py --model small_inception_no_bn --dataset cifar10
```

## Advanced Usage

### Using W&B for Logging

```bash
python train.py --use_wandb --wandb_project zhang-generalization
```

### Custom Learning Rate Schedule

```bash
# Step decay (drops at epochs 60, 80)
python train.py --lr_schedule step

# Cosine annealing
python train.py --lr_schedule cosine
```

### Full Example: Replicate Table 1 Row

```bash
# "Inception" row from Table 1
python train.py \
  --model small_inception \
  --dataset cifar10 \
  --batch_size 128 \
  --num_epochs 100 \
  --lr 0.01 \
  --momentum 0.9 \
  --optimizer sgd \
  --use_wandb \
  --wandb_project zhang-replication
```

## Configuration System

The `ExperimentConfig` class in `src/config.py` provides a clean interface for all paper toggles:

```python
from src.config import ExperimentConfig, get_baseline_config, get_random_labels_config

# Baseline experiment
config = get_baseline_config()

# Random labels experiment
config = get_random_labels_config()

# Custom configuration
config = ExperimentConfig(
    model_name='small_inception',
    dataset='cifar10',
    randomization='random_labels',
    weight_decay=0.0005,
    random_crop=True,
    augment_flip_rotate=False,
    batch_size=128,
    num_epochs=100,
)
```

## Project Structure

```
.
├── src/
│   ├── models/              # Neural network architectures
│   │   ├── small_inception.py
│   │   ├── small_alexnet.py
│   │   ├── mlp.py
│   │   └── utils.py
│   ├── data/                # Data loading and preprocessing
│   │   ├── transforms.py    # Per-image whitening, augmentation
│   │   └── datasets.py      # Randomization wrappers
│   └── config.py            # Experiment configuration
├── train.py                 # Main training script
├── test_zhang_models.py     # Model tests
├── test_data_pipeline.py    # Data pipeline tests
└── requirements.txt
```

## Key Implementation Details

### Per-Image Whitening
```python
# Subtract mean, divide by adjusted std (epsilon=1e-8)
mean = image.mean()
std = image.std()
whitened = (image - mean) / max(std, 1e-8)
```

### Consistent Random Labels
```python
# Labels are generated once and stored
# Same index always returns same random label
self.random_labels = rng.randint(0, num_classes, len(dataset))
```

### Weight Decay in Optimizer
```python
# Built-in PyTorch SGD weight decay (L2 regularization)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,  # Paper toggle
)
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{zhang2017understanding,
  title={Understanding deep learning requires rethinking generalization},
  author={Zhang, Chiyuan and Bengio, Samy and Hardt, Moritz and Recht, Benjamin and Vinyals, Oriol},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```

## Notes

- **Dropout**: Not implemented as the paper states "Only the Inception V3 for ImageNet uses dropout in our experiments." Since we're focusing on CIFAR-10/MNIST, dropout is not needed.
  
- **Weight Decay Defaults**: The paper mentions "default coefficient for each model" (Table 4) but doesn't specify exact values. We use reasonable defaults (0.0005 for CNNs, 0.0001 for MLPs) that can be overridden via config.

- **Input Shapes**: All models support variable input sizes via adaptive pooling (Inception) or dynamic flattened size computation (AlexNet, MLPs).

## License

MIT License - see LICENSE file for details.
