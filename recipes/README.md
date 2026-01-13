# Experiment Recipes for Zhang et al. 2017

This folder contains YAML configuration files for all CIFAR10 experiments from the paper "Understanding deep learning requires re-thinking generalization" by Zhang et al. (2017).

## 📁 Folder Structure

```
recipes/
├── baseline/           # Baseline experiments with true labels
├── randomization/      # Randomization tests (random labels, pixels, etc.)
├── regularization/     # Explicit regularization experiments
├── ablation/          # Ablation studies (regularization on random labels)
└── README.md          # This file
```

## 🎯 Experiment Categories

### 1. Baseline Experiments (`baseline/`)

Standard training on CIFAR10 with true labels and no explicit regularization.

| Recipe | Model | Description |
|--------|-------|-------------|
| `inception_baseline.yaml` | Inception | Baseline Inception (Table 1) |
| `alexnet_baseline.yaml` | Alexnet | Baseline Alexnet (Table 1) |
| `mlp_3x512_baseline.yaml` | MLP 3x512 | 3-layer MLP with 512 units (Table 1) |
| `mlp_1x512_baseline.yaml` | MLP 1x512 | 1-layer MLP with 512 units (Table 1) |

**Expected Results:**
- High test accuracy (~70-80% for CNNs, ~50-60% for MLPs)
- Good generalization

### 2. Randomization Experiments (`randomization/`)

Tests the central finding: **Deep neural networks easily fit random labels.**

#### Random Labels (Complete Label Randomization)
| Recipe | Model | Description |
|--------|-------|-------------|
| `inception_random_labels.yaml` | Inception | Random labels (Table 1) |
| `alexnet_random_labels.yaml` | Alexnet | Random labels (Table 1) |
| `mlp_3x512_random_labels.yaml` | MLP 3x512 | Random labels (Table 1) |
| `mlp_1x512_random_labels.yaml` | MLP 1x512 | Random labels (Table 1) |

**Expected Results:**
- 0% training error (100% training accuracy)
- ~10% test accuracy (random chance)

#### Pixel Randomization
| Recipe | Model | Description |
|--------|-------|-------------|
| `inception_gaussian_pixels.yaml` | Inception | Replace images with Gaussian noise |
| `inception_shuffled_pixels.yaml` | Inception | Shuffle pixels within each image |

**Expected Results:**
- Networks can still fit the data with 0% training error
- Shows that CNNs can fit pure noise

#### Partial Corruption (Interpolating Noise Levels)
| Recipe | Model | Corruption | Description |
|--------|-------|------------|-------------|
| `inception_partial_corrupt_20.yaml` | Inception | 20% | Corrupt 20% of labels |
| `inception_partial_corrupt_50.yaml` | Inception | 50% | Corrupt 50% of labels |
| `inception_partial_corrupt_80.yaml` | Inception | 80% | Corrupt 80% of labels |

**Expected Results:**
- Steady deterioration of test accuracy as noise increases
- Networks capture remaining signal while fitting noisy part

### 3. Regularization Experiments (`regularization/`)

Tests the effect of explicit regularization on true labels.

| Recipe | Regularizers | Description |
|--------|--------------|-------------|
| `inception_weight_decay.yaml` | Weight decay | L2 regularization only |
| `inception_random_crop.yaml` | Random crop | Data augmentation only |
| `inception_augmentation.yaml` | Flip + rotate | Full augmentation (Appendix E) |
| `inception_all_regularizers.yaml` | WD + crop | Combined regularization |

**Expected Results:**
- Regularization improves test accuracy
- But not necessary for reasonable generalization

### 4. Ablation Studies (`ablation/`)

Tests whether explicit regularization prevents fitting random labels (Appendix E, Table 4).

**Key Finding:** *Explicit regularization is neither necessary nor sufficient for controlling generalization error.*

| Recipe | Model | Regularizer | Expected Result |
|--------|-------|-------------|-----------------|
| `inception_random_labels_weight_decay.yaml` | Inception | Weight decay | 100% train acc |
| `inception_random_labels_random_crop.yaml` | Inception | Random crop | 99.93% train acc |
| `inception_random_labels_augmentation.yaml` | Inception | Flip + rotate | 99.28% train acc |
| `alexnet_random_labels_weight_decay.yaml` | Alexnet | Weight decay | Failed to converge |
| `mlp_3x512_random_labels_weight_decay.yaml` | MLP 3x512 | Weight decay | 100% train acc |
| `mlp_1x512_random_labels_weight_decay.yaml` | MLP 1x512 | Weight decay | 99.21% train acc |

## 🚀 Usage

### Running a Single Experiment

```bash
# Using the recipe directly
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml

# Enable W&B logging
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --use_wandb
```

### Running Multiple Experiments

```bash
# Run all baseline experiments
for recipe in recipes/baseline/*.yaml; do
    ./run_experiment.sh --config "$recipe"
done

# Run all randomization experiments
for recipe in recipes/randomization/*.yaml; do
    ./run_experiment.sh --config "$recipe"
done
```

### Modifying Recipes

You can create custom recipes by copying and modifying existing ones:

```bash
cp recipes/baseline/inception_baseline.yaml recipes/my_experiment.yaml
# Edit my_experiment.yaml with your changes
./run_experiment.sh --config recipes/my_experiment.yaml
```

## 📊 W&B Integration

All recipes are configured for the W&B project:
- **Project:** `FOLI-Project`
- **Entity:** `alirezasakhaeirad`
- **URL:** https://wandb.ai/alirezasakhaeirad/FOLI-Project

To enable W&B logging:
1. Set your API key: `export WANDB_API_KEY=your_key_here`
2. Add `--use_wandb` flag when running experiments

## 🔑 Key Configuration Parameters

### Model Selection
```yaml
model_name: small_inception  # or small_alexnet, mlp_1x512, mlp_3x512
```

### Randomization Types
```yaml
randomization: null              # No randomization (true labels)
randomization: random_labels     # Complete label randomization
randomization: partial_corrupt   # Partial label corruption
randomization: shuffled_pixels   # Shuffle pixels within images
randomization: gaussian_pixels   # Replace with Gaussian noise
```

### Regularization
```yaml
weight_decay: 0.0005            # L2 regularization
random_crop: true               # Random cropping augmentation
augment_flip_rotate: true       # Flip + rotate augmentation
```

### Training Hyperparameters
```yaml
batch_size: 128
num_epochs: 100
learning_rate: 0.01
momentum: 0.9
optimizer: sgd                  # or adam
```

## 📖 Paper Reference

Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires re-thinking generalization. In International Conference on Learning Representations (ICLR).

## 🎓 Key Takeaways from the Paper

1. **Deep neural networks easily fit random labels** - achieving 0% training error
2. **Optimization remains easy** even on random labels
3. **Explicit regularization** (weight decay, dropout, data augmentation) is:
   - Not necessary for generalization
   - Not sufficient to prevent overfitting to random labels
4. **Traditional complexity measures** (VC dimension, Rademacher complexity) fail to explain generalization
5. **Networks can fit pure noise** - even replacing images with Gaussian noise

## 🔬 Reproducing Paper Results

To reproduce the main results from Table 1:

```bash
# 1. Baseline (true labels, no regularization)
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml

# 2. Random labels (no regularization)
./run_experiment.sh --config recipes/randomization/inception_random_labels.yaml

# 3. Compare: baseline should generalize well, random labels should not
```

Expected output:
- **Baseline:** ~75-80% test accuracy
- **Random labels:** ~10% test accuracy (random chance), but 100% training accuracy

This demonstrates the paper's central finding: the same model with the same hyperparameters can either generalize well or completely fail, depending only on the data labels.

