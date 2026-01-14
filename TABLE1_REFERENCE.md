# Table 1 Experiments - Quick Reference

This document provides a complete reference for all 19 experiments from Table 1 of paper 1611.03530v2.

## Quick Start

### View all experiments (dry-run):
```bash
./generate_table1_experiments.sh --dry-run
```

### Run all experiments:
```bash
./generate_table1_experiments.sh
```

### Run individual experiment groups:
```bash
# Using the existing run_all_experiments.sh
./run_all_experiments.sh --table1
```

## Complete Experiment List

### Small Inception (1,649,402 params) - 5 experiments

| # | Random Crop | Weight Decay | Randomization | Train Acc | Test Acc | Command |
|---|-------------|--------------|---------------|-----------|----------|---------|
| 1 | yes | 0.0005 | none | 100.0 | 89.05 | `--model small_inception --random_crop --weight_decay 0.0005` |
| 2 | yes | no | none | 100.0 | 89.31 | `--model small_inception --random_crop` |
| 3 | no | 0.0005 | none | 100.0 | 86.03 | `--model small_inception --weight_decay 0.0005` |
| 4 | no | no | none | 100.0 | 85.75 | `--model small_inception` |
| 5 | no | no | random_labels | 100.0 | 9.78 | `--model small_inception --randomization random_labels` |

### Inception w/o BatchNorm (1,649,402 params) - 3 experiments

| # | Random Crop | Weight Decay | Randomization | Train Acc | Test Acc | Command |
|---|-------------|--------------|---------------|-----------|----------|---------|
| 6 | no | 0.0005 | none | 100.0 | 83.00 | `--model small_inception_no_bn --weight_decay 0.0005` |
| 7 | no | no | none | 100.0 | 82.00 | `--model small_inception_no_bn` |
| 8 | no | no | random_labels | 100.0 | 10.12 | `--model small_inception_no_bn --randomization random_labels` |

### Small AlexNet (1,387,786 params) - 5 experiments

| # | Random Crop | Weight Decay | Randomization | Train Acc | Test Acc | Command |
|---|-------------|--------------|---------------|-----------|----------|---------|
| 9 | yes | 0.0005 | none | 99.90 | 81.22 | `--model small_alexnet --random_crop --weight_decay 0.0005` |
| 10 | yes | no | none | 99.82 | 79.66 | `--model small_alexnet --random_crop` |
| 11 | no | 0.0005 | none | 100.0 | 77.36 | `--model small_alexnet --weight_decay 0.0005` |
| 12 | no | no | none | 100.0 | 76.07 | `--model small_alexnet` |
| 13 | no | no | random_labels | 99.82 | 9.86 | `--model small_alexnet --randomization random_labels` |

### MLP 3×512 (1,735,178 params) - 3 experiments

| # | Random Crop | Weight Decay | Randomization | Train Acc | Test Acc | Command |
|---|-------------|--------------|---------------|-----------|----------|---------|
| 14 | no | 0.0001 | none | 100.0 | 53.35 | `--model mlp_3x512 --weight_decay 0.0001` |
| 15 | no | no | none | 100.0 | 52.39 | `--model mlp_3x512` |
| 16 | no | no | random_labels | 100.0 | 10.48 | `--model mlp_3x512 --randomization random_labels` |

### MLP 1×512 (1,209,866 params) - 3 experiments

| # | Random Crop | Weight Decay | Randomization | Train Acc | Test Acc | Command |
|---|-------------|--------------|---------------|-----------|----------|---------|
| 17 | no | 0.0001 | none | 99.80 | 50.39 | `--model mlp_1x512 --weight_decay 0.0001` |
| 18 | no | no | none | 100.0 | 50.51 | `--model mlp_1x512` |
| 19 | no | no | random_labels | 99.34 | 10.61 | `--model mlp_1x512 --randomization random_labels` |

## Running Individual Experiments

Each experiment can be run individually using:

```bash
python train.py --model <model> --dataset cifar10 --num_epochs 100 --use_wandb [OPTIONS]
```

### Example Commands

**Inception baseline (no regularization):**
```bash
python train.py --model small_inception --dataset cifar10 --num_epochs 100 --use_wandb
```

**Inception with full regularization:**
```bash
python train.py --model small_inception --dataset cifar10 --num_epochs 100 --use_wandb --random_crop --weight_decay 0.0005
```

**AlexNet with random labels:**
```bash
python train.py --model small_alexnet --dataset cifar10 --num_epochs 100 --use_wandb --randomization random_labels
```

**MLP 3×512 with weight decay:**
```bash
python train.py --model mlp_3x512 --dataset cifar10 --num_epochs 100 --use_wandb --weight_decay 0.0001
```

## Key Observations from Table 1

1. **Random Crop is most effective**: Inception achieves 89.31% test accuracy with random crop alone vs 85.75% baseline
2. **Weight decay helps less**: Adding weight decay to random crop only improves from 89.31% to 89.05%
3. **BatchNorm matters**: Removing BatchNorm drops test accuracy from 82.00% to ~82.00% (Inception No BN)
4. **All models fit random labels**: Every model achieves ~100% training accuracy on random labels
5. **Random labels → ~10% test accuracy**: All models achieve ~10% test accuracy (random chance) on random labels
6. **MLPs have lower capacity**: MLP models achieve ~50% test accuracy vs ~85% for CNNs

## Notes

- All experiments use CIFAR-10 dataset
- All experiments run for 100 epochs
- Weight decay values:
  - CNNs (Inception, AlexNet): 0.0005
  - MLPs: 0.0001
- Random crop is only used with CNN models (not MLPs)
- Results are logged to W&B project: `FOLI-Project`

## Output Locations

- **Checkpoints**: `./checkpoints/<model>_<dataset>_checkpoint.pth`
- **Best model**: `./checkpoints/<model>_<dataset>_best.pth`
- **Results**: `./results/<model>_<dataset>_<randomization>_<timestamp>.json`
- **W&B Dashboard**: https://wandb.ai/alirezasakhaeirad/FOLI-Project
