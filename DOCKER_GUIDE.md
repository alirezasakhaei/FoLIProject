# Docker Deployment Guide

## Building the Docker Image

```bash
docker build -t zhang-experiments .
```

## Running Experiments

### Prerequisites

Set your Weights & Biases API key:
```bash
export WANDB_API_KEY=your_api_key_here
```

### Using Recipe Files (Recommended)

Run experiments using pre-configured YAML recipes:

```bash
# Baseline experiment
docker run --rm \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/data:/workspace/data \
  --gpus all \
  zhang-experiments --config recipes/baseline.yaml

# Random labels experiment
docker run --rm \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/data:/workspace/data \
  --gpus all \
  zhang-experiments --config recipes/random_labels.yaml

# Regularized experiment
docker run --rm \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/data:/workspace/data \
  --gpus all \
  zhang-experiments --config recipes/regularized.yaml
```

### Using Direct CLI Arguments

You can also run experiments with direct command-line arguments:

```bash
docker run --rm \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/data:/workspace/data \
  --gpus all \
  zhang-experiments \
  --model inception \
  --dataset cifar10 \
  --num_epochs 100 \
  --lr 0.01
```

## All Available Recipe Configurations

### Basic Experiments

1. **baseline.yaml** - Baseline Small Inception on CIFAR-10
2. **random_labels.yaml** - Random labels experiment
3. **regularized.yaml** - With weight decay and random crop
4. **alexnet_baseline.yaml** - AlexNet architecture
5. **mlp_3x512_baseline.yaml** - MLP architecture

### Randomization Experiments (Figure 1)

6. **shuffled_pixels.yaml** - Fixed pixel permutation
7. **random_pixels.yaml** - Independent permutation per image
8. **gaussian_pixels.yaml** - Gaussian noise images
9. **partial_corrupt_50.yaml** - 50% label corruption

## Complete Command Reference

### Model Selection
```bash
--model {inception, inception_no_bn, small_alexnet, mlp_1x512, mlp_3x512}
```

### Dataset Selection
```bash
--dataset {cifar10, mnist}
--data_root ./data
```

### Randomization Options
```bash
# Random labels (full randomization)
--randomization random_labels

# Partial label corruption
--randomization partial_corrupt --corruption_prob 0.5

# Shuffled pixels (fixed permutation)
--randomization shuffled_pixels

# Random pixels (independent permutation per image)
--randomization random_pixels

# Gaussian noise
--randomization gaussian_pixels

# Control randomization seed
--randomization_seed 42
```

### Regularization Options
```bash
# Weight decay (L2 regularization)
--weight_decay 0.0005

# Random crop augmentation
--random_crop

# Flip + rotate augmentation
--augment_flip_rotate
```

### Training Hyperparameters
```bash
--batch_size 128
--num_epochs 100
--lr 0.01
--momentum 0.9
--optimizer {sgd, adam}
--lr_schedule {step, cosine}
```

### Logging and Checkpointing
```bash
--use_wandb                    # Enable W&B logging
--wandb_project zhang-generalization
--wandb_entity your_entity
--save_dir ./checkpoints       # Checkpoint directory
--log_interval 10              # Log every N batches
```

### Other Options
```bash
--seed 42                      # Random seed
--device {cuda, cpu, mps}      # Device selection
```

## Example Commands for Paper Experiments

### Figure 1a: Learning Curves Comparison

```bash
# True labels
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/results:/workspace/results -v $(pwd)/data:/workspace/data --gpus all zhang-experiments --config recipes/baseline.yaml

# Random labels
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/results:/workspace/results -v $(pwd)/data:/workspace/data --gpus all zhang-experiments --config recipes/random_labels.yaml

# Shuffled pixels
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/results:/workspace/results -v $(pwd)/data:/workspace/data --gpus all zhang-experiments --config recipes/shuffled_pixels.yaml

# Random pixels
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/results:/workspace/results -v $(pwd)/data:/workspace/data --gpus all zhang-experiments --config recipes/random_pixels.yaml

# Gaussian noise
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/results:/workspace/results -v $(pwd)/data:/workspace/data --gpus all zhang-experiments --config recipes/gaussian_pixels.yaml
```

### Figure 1b/1c: Corruption Sweep

```bash
# Sweep corruption probability from 0 to 1
for p in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/results:/workspace/results -v $(pwd)/data:/workspace/data --gpus all zhang-experiments --model inception --randomization partial_corrupt --corruption_prob $p
done
```

### Table 1: Regularization Grid

```bash
# Inception: no reg
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/results:/workspace/results -v $(pwd)/data:/workspace/data --gpus all zhang-experiments --model inception

# Inception: weight decay only
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/results:/workspace/results -v $(pwd)/data:/workspace/data --gpus all zhang-experiments --model inception --weight_decay 0.0005

# Inception: random crop only
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/results:/workspace/results -v $(pwd)/data:/workspace/data --gpus all zhang-experiments --model inception --random_crop

# Inception: both
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/results:/workspace/results -v $(pwd)/data:/workspace/data --gpus all zhang-experiments --config recipes/regularized.yaml
```

## RunAI Deployment

For RunAI-based clusters, use the following command template:

```bash
runai submit zhang-exp-baseline \
  --image zhang-experiments:latest \
  --gpu 1 \
  --pvc data:/workspace/data \
  --pvc checkpoints:/workspace/checkpoints \
  --pvc results:/workspace/results \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -- --config recipes/baseline.yaml
```

### RunAI Job Examples

```bash
# Submit baseline experiment
runai submit zhang-baseline \
  --image zhang-experiments:latest \
  --gpu 1 \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -- --config recipes/baseline.yaml

# Submit random labels experiment
runai submit zhang-random-labels \
  --image zhang-experiments:latest \
  --gpu 1 \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -- --config recipes/random_labels.yaml

# Submit with direct arguments
runai submit zhang-custom \
  --image zhang-experiments:latest \
  --gpu 1 \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -- --model small_alexnet --dataset cifar10 --weight_decay 0.0005 --random_crop
```

## Directory Structure

```
/workspace/
├── checkpoints/          # Model checkpoints (auto-resumable)
│   ├── inception_cifar10_checkpoint.pth
│   └── inception_cifar10_best.pth
├── results/              # Experiment results (JSON files)
│   └── inception_cifar10_baseline_20260113_221600.json
├── data/                 # Dataset cache
│   └── cifar-10-batches-py/
└── recipes/              # Experiment configurations
    ├── baseline.yaml
    ├── random_labels.yaml
    └── ...
```

## Checkpoint Resumption

The training script automatically:
- Saves checkpoints after every epoch
- Resumes from the last checkpoint if it exists
- Saves the best model separately based on test accuracy

To restart a failed experiment, simply run the same command again. The script will automatically detect and load the checkpoint.

## Results Format

Results are saved as JSON files with the following structure:

```json
{
  "config": {
    "model_name": "inception",
    "dataset": "cifar10",
    "randomization": "random_labels",
    ...
  },
  "metrics": {
    "epochs": [1, 2, 3, ...],
    "train_loss": [2.3, 2.1, ...],
    "train_acc": [10.5, 15.2, ...],
    "test_loss": [2.3, 2.2, ...],
    "test_acc": [10.0, 12.5, ...],
    "best_test_acc": 85.5,
    "final_train_acc": 99.8,
    "final_test_acc": 85.5
  },
  "timestamp": "20260113_221600"
}
```

These JSON files can be easily parsed for plotting and analysis.
