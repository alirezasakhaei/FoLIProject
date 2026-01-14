# All Experiment Commands Reference

This document lists all possible experiment configurations that can be run.

## Recipe-Based Commands (Recommended)

All recipes are in the `recipes/` directory and can be used with:
```bash
./run_experiment.sh --config recipes/<recipe_name>.yaml
```

Or with Docker:
```bash
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/data:/workspace/data \
  --gpus all \
  zhang-experiments --config recipes/<recipe_name>.yaml
```

### Available Recipes

**See `recipes/README.md` for complete documentation of all experiments!**

#### Table 1: Main Results (19 experiments total)

**Small Inception (5 configs):**
1. **baseline.yaml** - No regularization
2. **inception_weight_decay.yaml** - Weight decay only
3. **inception_random_crop.yaml** - Random crop only
4. **regularized.yaml** - Weight decay + random crop
5. **inception_random_labels.yaml** - Random labels

**Small Inception No BN (3 configs):**
6. **inception_no_bn_baseline.yaml** - No regularization
7. **inception_no_bn_weight_decay.yaml** - Weight decay
8. **inception_no_bn_random_labels.yaml** - Random labels

**Small AlexNet (5 configs):**
9. **alexnet_baseline.yaml** - No regularization
10. **alexnet_weight_decay.yaml** - Weight decay only
11. **alexnet_random_crop.yaml** - Random crop only
12. **alexnet_regularized.yaml** - Weight decay + random crop
13. **alexnet_random_labels.yaml** - Random labels

**MLP 3x512 (3 configs):**
14. **mlp_3x512_baseline.yaml** - No regularization
15. **mlp_3x512_weight_decay.yaml** - Weight decay
16. **mlp_3x512_random_labels.yaml** - Random labels

**MLP 1x512 (3 configs):**
17. **mlp_1x512_baseline.yaml** - No regularization
18. **mlp_1x512_weight_decay.yaml** - Weight decay
19. **mlp_1x512_random_labels.yaml** - Random labels

#### Figure 1a: Learning Curves (5 experiments)

1. **baseline.yaml** - True labels
2. **random_labels.yaml** - Random labels
3. **shuffled_pixels.yaml** - Fixed pixel permutation
4. **random_pixels.yaml** - Random pixel permutation
5. **gaussian_pixels.yaml** - Gaussian noise

#### Figure 1b/1c: Corruption Sweep

Located in `corruption_sweep/` subdirectory:
- **corruption_sweep/inception_corrupt_10.yaml** - 10% corruption
- **corruption_sweep/inception_corrupt_20.yaml** - 20% corruption
- **corruption_sweep/inception_corrupt_80.yaml** - 80% corruption
- **partial_corrupt_50.yaml** - 50% corruption

#### Appendix E / Table 4: Stress Tests (6 experiments)

Located in `stress_tests/` subdirectory:
- **stress_tests/inception_random_labels_wd.yaml** - Inception + random labels + weight decay
- **stress_tests/inception_random_labels_crop.yaml** - Inception + random labels + random crop
- **stress_tests/inception_random_labels_aug.yaml** - Inception + random labels + augmentation
- **stress_tests/alexnet_random_labels_wd.yaml** - AlexNet + random labels + weight decay
- **stress_tests/mlp_3x512_random_labels_wd.yaml** - MLP 3x512 + random labels + weight decay
- **stress_tests/mlp_1x512_random_labels_wd.yaml** - MLP 1x512 + random labels + weight decay

## Direct CLI Commands

### Model Variants

```bash
# Small Inception (default)
--model inception

# Small Inception without BatchNorm
--model inception_no_bn

# Small AlexNet
--model small_alexnet

# MLP with 1 hidden layer of 512 units
--model mlp_1x512

# MLP with 3 hidden layers of 512 units each
--model mlp_3x512
```

### Dataset Variants

```bash
# CIFAR-10 (default)
--dataset cifar10

# MNIST
--dataset mnist
```

### Randomization Modes

```bash
# No randomization (baseline)
# (no flag needed)

# Random labels
--randomization random_labels

# Partial label corruption (specify probability)
--randomization partial_corrupt --corruption_prob 0.5

# Shuffled pixels (fixed permutation)
--randomization shuffled_pixels

# Random pixels (independent permutation per image)
--randomization random_pixels

# Gaussian noise
--randomization gaussian_pixels
```

### Regularization Combinations

```bash
# No regularization (default)
# (no flags needed)

# Weight decay only
--weight_decay 0.0005

# Random crop only
--random_crop

# Augmentation (flip + rotate) only
--augment_flip_rotate

# Weight decay + random crop
--weight_decay 0.0005 --random_crop

# Weight decay + augmentation
--weight_decay 0.0005 --augment_flip_rotate

# All regularization
--weight_decay 0.0005 --random_crop --augment_flip_rotate
```

## Complete Experiment Matrix for Table 1

### Small Inception (4 regularization configs + random labels)

```bash
# 1. No regularization
--model inception

# 2. Weight decay only
--model inception --weight_decay 0.0005

# 3. Random crop only
--model inception --random_crop

# 4. Weight decay + random crop
--model inception --weight_decay 0.0005 --random_crop

# 5. Random labels (no reg)
--model inception --randomization random_labels
```

### Small Inception No BN (2 configs + random labels)

```bash
# 1. Weight decay
--model inception_no_bn --weight_decay 0.0005

# 2. No regularization
--model inception_no_bn

# 3. Random labels
--model inception_no_bn --randomization random_labels
```

### Small AlexNet (4 configs + random labels)

```bash
# 1. No regularization
--model small_alexnet

# 2. Weight decay only
--model small_alexnet --weight_decay 0.0005

# 3. Random crop only
--model small_alexnet --random_crop

# 4. Weight decay + random crop
--model small_alexnet --weight_decay 0.0005 --random_crop

# 5. Random labels
--model small_alexnet --randomization random_labels
```

### MLP 3x512 (2 configs + random labels)

```bash
# 1. Weight decay
--model mlp_3x512 --weight_decay 0.0001

# 2. No regularization
--model mlp_3x512

# 3. Random labels
--model mlp_3x512 --randomization random_labels
```

### MLP 1x512 (2 configs + random labels)

```bash
# 1. Weight decay
--model mlp_1x512 --weight_decay 0.0001

# 2. No regularization
--model mlp_1x512

# 3. Random labels
--model mlp_1x512 --randomization random_labels
```

## Figure 1a: Learning Curves

```bash
# True labels
--model inception

# Random labels
--model inception --randomization random_labels

# Shuffled pixels
--model inception --randomization shuffled_pixels

# Random pixels
--model inception --randomization random_pixels

# Gaussian noise
--model inception --randomization gaussian_pixels
```

## Figure 1b/1c: Corruption Sweep

For each model (inception, small_alexnet, mlp_1x512):

```bash
# Corruption probabilities: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
--model <model_name> --randomization partial_corrupt --corruption_prob <p>
```

## Appendix E / Table 4: Explicit Regularization Stress Tests

### Random labels + weight decay

```bash
# Inception
--model inception --randomization random_labels --weight_decay 0.0005

# AlexNet
--model small_alexnet --randomization random_labels --weight_decay 0.0005

# MLP 3x512
--model mlp_3x512 --randomization random_labels --weight_decay 0.0001

# MLP 1x512
--model mlp_1x512 --randomization random_labels --weight_decay 0.0001
```

### Random labels + random cropping

```bash
--model inception --randomization random_labels --random_crop
```

### Random labels + augmentation

```bash
--model inception --randomization random_labels --augment_flip_rotate
```

## Training Hyperparameter Variations

```bash
# Different learning rates
--lr 0.001
--lr 0.01   # default
--lr 0.1

# Different batch sizes
--batch_size 64
--batch_size 128  # default
--batch_size 256

# Different optimizers
--optimizer sgd   # default
--optimizer adam

# Learning rate schedules
--lr_schedule step     # MultiStepLR at epochs 60, 80
--lr_schedule cosine   # Cosine annealing

# Different number of epochs
--num_epochs 50
--num_epochs 100  # default
--num_epochs 200
```

## Full Command Examples

### Example 1: Baseline with W&B logging
```bash
python train.py \
  --model inception \
  --dataset cifar10 \
  --num_epochs 100 \
  --lr 0.01 \
  --use_wandb
```

### Example 2: Random labels experiment
```bash
python train.py \
  --model inception \
  --dataset cifar10 \
  --randomization random_labels \
  --num_epochs 100 \
  --use_wandb
```

### Example 3: Regularized training
```bash
python train.py \
  --model inception \
  --dataset cifar10 \
  --weight_decay 0.0005 \
  --random_crop \
  --num_epochs 100 \
  --use_wandb
```

### Example 4: Corruption sweep (single point)
```bash
python train.py \
  --model inception \
  --dataset cifar10 \
  --randomization partial_corrupt \
  --corruption_prob 0.5 \
  --num_epochs 100 \
  --use_wandb
```

### Example 5: Using YAML config
```bash
python train.py --config recipes/baseline.yaml --use_wandb
```

## Bash Script Usage

The `run_experiment.sh` script automatically handles WANDB_API_KEY:

```bash
# With config file
./run_experiment.sh --config recipes/baseline.yaml

# With direct arguments
./run_experiment.sh --model inception --dataset cifar10 --num_epochs 100
```

## Environment Variables

```bash
# Required for W&B logging
export WANDB_API_KEY=your_api_key

# Optional: W&B project and entity
export WANDB_PROJECT=zhang-generalization
export WANDB_ENTITY=your_entity
```

## Output Locations

- **Checkpoints**: `./checkpoints/<model>_<dataset>_checkpoint.pth`
- **Best model**: `./checkpoints/<model>_<dataset>_best.pth`
- **Results**: `./results/<model>_<dataset>_<randomization>_<timestamp>.json`
- **Data**: `./data/` (auto-downloaded)

## Notes

1. All experiments automatically resume from checkpoints if they exist
2. Results are saved as JSON files with complete metrics history
3. W&B logging is optional but recommended for tracking experiments
4. The script creates directories automatically if they don't exist

## Running All Experiments Systematically

Use the master script `run_all_experiments.sh` to run experiments by category:

```bash
# Make executable (if not already)
chmod +x run_all_experiments.sh

# Run all Table 1 experiments (19 configs)
./run_all_experiments.sh --table1

# Run all Figure 1a experiments (5 configs)
./run_all_experiments.sh --figure1a

# Run all stress test experiments (Appendix E)
./run_all_experiments.sh --stress-tests

# Run corruption sweep experiments (Figure 1b/1c)
./run_all_experiments.sh --corruption-sweep

# Run EVERYTHING
./run_all_experiments.sh --all
```

The master script provides:
- Colored output for easy tracking
- Automatic sequential execution
- Progress indicators
- Organized by experiment type

## W&B Configuration

All recipes are configured to log to:
- **Project**: `FOLI-Project`
- **Entity**: `alirezasakhaeirad`

Set your W&B API key:
```bash
export WANDB_API_KEY=your_api_key
```
