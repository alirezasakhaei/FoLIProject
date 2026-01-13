# Complete Index of CIFAR10 Experiments

This document provides a complete index of all experiments from Zhang et al. 2017 paper, organized by category.

## Summary Statistics

- **Total Recipes:** 29
- **Baseline Experiments:** 4
- **Randomization Experiments:** 9
- **Regularization Experiments:** 4
- **Ablation Studies:** 6

---

## 1. Baseline Experiments (4 recipes)

Training on true labels without explicit regularization.

| # | Recipe File | Model | Regularization | Expected Train Acc | Expected Test Acc |
|---|-------------|-------|----------------|-------------------|-------------------|
| 1 | `baseline/inception_baseline.yaml` | Inception | None | ~99% | ~75-80% |
| 2 | `baseline/alexnet_baseline.yaml` | Alexnet | None | ~99% | ~70-75% |
| 3 | `baseline/mlp_3x512_baseline.yaml` | MLP 3x512 | None | ~95% | ~55-60% |
| 4 | `baseline/mlp_1x512_baseline.yaml` | MLP 1x512 | None | ~90% | ~50-55% |

**Paper Reference:** Table 1 (True labels column)

---

## 2. Randomization Experiments (9 recipes)

### 2.1 Random Labels (4 recipes)

Complete label randomization - the central experiment of the paper.

| # | Recipe File | Model | Expected Train Acc | Expected Test Acc |
|---|-------------|-------|-------------------|-------------------|
| 5 | `randomization/inception_random_labels.yaml` | Inception | 100% | ~10% |
| 6 | `randomization/alexnet_random_labels.yaml` | Alexnet | 100% | ~10% |
| 7 | `randomization/mlp_3x512_random_labels.yaml` | MLP 3x512 | 100% | ~10% |
| 8 | `randomization/mlp_1x512_random_labels.yaml` | MLP 1x512 | ~99% | ~10% |

**Paper Reference:** Table 1 (Random labels column)

**Key Finding:** Networks achieve 0% training error on random labels, but test accuracy is at chance level.

### 2.2 Pixel Randomization (2 recipes)

Testing if CNNs can fit pure noise or shuffled pixels.

| # | Recipe File | Randomization Type | Expected Train Acc | Expected Test Acc |
|---|-------------|-------------------|-------------------|-------------------|
| 9 | `randomization/inception_gaussian_pixels.yaml` | Gaussian noise | 100% | ~10% |
| 10 | `randomization/inception_shuffled_pixels.yaml` | Shuffled pixels | 100% | ~10% |

**Paper Reference:** Section 1.1 (Randomization tests)

**Key Finding:** CNNs can fit completely random noise, showing massive capacity for memorization.

### 2.3 Partial Corruption (3 recipes)

Interpolating between true labels and random labels.

| # | Recipe File | Corruption % | Expected Train Acc | Expected Test Acc |
|---|-------------|--------------|-------------------|-------------------|
| 11 | `randomization/inception_partial_corrupt_20.yaml` | 20% | ~99% | ~65-70% |
| 12 | `randomization/inception_partial_corrupt_50.yaml` | 50% | ~99% | ~45-50% |
| 13 | `randomization/inception_partial_corrupt_80.yaml` | 80% | ~99% | ~20-25% |

**Paper Reference:** Section 1.1 (Randomization tests)

**Key Finding:** Steady deterioration of generalization as noise increases. Networks capture remaining signal while fitting noisy part.

---

## 3. Regularization Experiments (4 recipes)

Testing the effect of explicit regularization on true labels.

| # | Recipe File | Regularizers | Expected Train Acc | Expected Test Acc |
|---|-------------|--------------|-------------------|-------------------|
| 14 | `regularization/inception_weight_decay.yaml` | Weight decay | ~98% | ~77-82% |
| 15 | `regularization/inception_random_crop.yaml` | Random crop | ~98% | ~78-83% |
| 16 | `regularization/inception_augmentation.yaml` | Flip + rotate | ~98% | ~79-84% |
| 17 | `regularization/inception_all_regularizers.yaml` | WD + crop | ~97% | ~80-85% |

**Paper Reference:** Table 1 (Regularization columns)

**Key Finding:** Regularization improves test accuracy but is not necessary for reasonable generalization.

---

## 4. Ablation Studies (6 recipes)

Testing if explicit regularization prevents fitting random labels.

| # | Recipe File | Model | Regularizer | Expected Train Acc | Expected Test Acc |
|---|-------------|-------|-------------|-------------------|-------------------|
| 18 | `ablation/inception_random_labels_weight_decay.yaml` | Inception | Weight decay | 100% | ~10% |
| 19 | `ablation/inception_random_labels_random_crop.yaml` | Inception | Random crop | 99.93% | ~10% |
| 20 | `ablation/inception_random_labels_augmentation.yaml` | Inception | Flip + rotate | 99.28% | ~10% |
| 21 | `ablation/alexnet_random_labels_weight_decay.yaml` | Alexnet | Weight decay | Failed* | ~10% |
| 22 | `ablation/mlp_3x512_random_labels_weight_decay.yaml` | MLP 3x512 | Weight decay | 100% | ~10% |
| 23 | `ablation/mlp_1x512_random_labels_weight_decay.yaml` | MLP 1x512 | Weight decay | 99.21% | ~10% |

**Paper Reference:** Appendix E, Table 4

**Key Finding:** Explicit regularization does NOT prevent networks from fitting random labels (except Alexnet which failed to converge).

*Note: Alexnet failed to converge with weight decay on random labels, as reported in Table 4.

---

## Quick Reference: Running Experiments

### Run a single experiment
```bash
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml
```

### Run all experiments in a category
```bash
./recipes/run_all_recipes.sh --category baseline
./recipes/run_all_recipes.sh --category randomization
./recipes/run_all_recipes.sh --category regularization
./recipes/run_all_recipes.sh --category ablation
```

### Run ALL experiments
```bash
./recipes/run_all_recipes.sh --category all
```

### Enable W&B logging
```bash
export WANDB_API_KEY=your_key_here
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --use_wandb
```

---

## Experiment Priority

If you have limited compute time, run experiments in this order:

### Priority 1: Core Findings (2 experiments, ~6-8 hours)
1. `baseline/inception_baseline.yaml` - Baseline performance
2. `randomization/inception_random_labels.yaml` - Central finding

### Priority 2: Model Comparison (6 experiments, ~18-24 hours)
3. `baseline/alexnet_baseline.yaml`
4. `baseline/mlp_3x512_baseline.yaml`
5. `randomization/alexnet_random_labels.yaml`
6. `randomization/mlp_3x512_random_labels.yaml`
7. `randomization/inception_gaussian_pixels.yaml`
8. `randomization/inception_shuffled_pixels.yaml`

### Priority 3: Regularization Effects (7 experiments, ~21-28 hours)
9. `regularization/inception_weight_decay.yaml`
10. `regularization/inception_random_crop.yaml`
11. `regularization/inception_all_regularizers.yaml`
12. `ablation/inception_random_labels_weight_decay.yaml`
13. `ablation/inception_random_labels_random_crop.yaml`
14. `randomization/inception_partial_corrupt_20.yaml`
15. `randomization/inception_partial_corrupt_50.yaml`

### Priority 4: Complete Reproduction (All 29 experiments, ~80-100 hours)
Run all remaining experiments for complete paper reproduction.

---

## Expected Runtime

Approximate runtime per experiment on a single GPU (V100/A100):
- **Inception models:** ~3-4 hours (100 epochs)
- **Alexnet models:** ~2-3 hours (100 epochs)
- **MLP models:** ~1-2 hours (100 epochs)

Total runtime for all 29 experiments: **~80-100 GPU hours**

---

## Validation Checklist

Use this checklist to verify your reproduction:

- [ ] Baseline Inception achieves >75% test accuracy
- [ ] Random labels Inception achieves 100% train accuracy
- [ ] Random labels Inception achieves ~10% test accuracy (chance)
- [ ] Gaussian pixels Inception achieves 100% train accuracy
- [ ] Partial corruption shows gradual test accuracy degradation
- [ ] Weight decay improves test accuracy on true labels
- [ ] Weight decay does NOT prevent fitting random labels
- [ ] Alexnet fails to converge with weight decay on random labels
- [ ] Augmentation achieves 99.28% train acc on random labels

---

## Paper Results Reference

### Table 1: Main Results (True vs Random Labels)

| Model | True Labels (Train/Test) | Random Labels (Train/Test) |
|-------|--------------------------|----------------------------|
| Inception | High/High | 100%/~10% |
| Alexnet | High/High | 100%/~10% |
| MLP 3x512 | High/Medium | 100%/~10% |
| MLP 1x512 | Medium/Medium | ~99%/~10% |

### Table 4 (Appendix E): Fitting Random Labels with Regularization

| Model | Regularizer | Training Accuracy |
|-------|-------------|-------------------|
| Inception | None | 100% |
| Inception | Weight decay | 100% |
| Inception | Random crop | 99.93% |
| Inception | Augmentation | 99.28% |
| Alexnet | Weight decay | Failed to converge |
| MLP 3x512 | Weight decay | 100% |
| MLP 1x512 | Weight decay | 99.21% |

---

## Notes

- All experiments use SGD with momentum (0.9) and learning rate 0.01
- Batch size is 128 for all experiments
- Default training is 100 epochs (150 for some augmentation experiments)
- Random seed is 42 for reproducibility
- CIFAR10 images are center-cropped to 28x28 (as per paper)

