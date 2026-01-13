# Complete Experiment Matrix

## Overview
- **Total Recipe Files**: 33
- **Main Recipes**: 24
- **Corruption Sweep**: 3 (expandable to 11)
- **Stress Tests**: 6

## Table 1: Complete Model × Regularization Matrix

| Model | No Reg | Weight Decay | Random Crop | WD + Crop | Random Labels |
|-------|--------|--------------|-------------|-----------|---------------|
| **Small Inception** | ✅ baseline.yaml | ✅ inception_weight_decay.yaml | ✅ inception_random_crop.yaml | ✅ regularized.yaml | ✅ inception_random_labels.yaml |
| **Inception No BN** | ✅ inception_no_bn_baseline.yaml | ✅ inception_no_bn_weight_decay.yaml | - | - | ✅ inception_no_bn_random_labels.yaml |
| **Small AlexNet** | ✅ alexnet_baseline.yaml | ✅ alexnet_weight_decay.yaml | ✅ alexnet_random_crop.yaml | ✅ alexnet_regularized.yaml | ✅ alexnet_random_labels.yaml |
| **MLP 3x512** | ✅ mlp_3x512_baseline.yaml | ✅ mlp_3x512_weight_decay.yaml | - | - | ✅ mlp_3x512_random_labels.yaml |
| **MLP 1x512** | ✅ mlp_1x512_baseline.yaml | ✅ mlp_1x512_weight_decay.yaml | - | - | ✅ mlp_1x512_random_labels.yaml |

**Total: 19 experiments**

## Figure 1a: Randomization Types (Small Inception)

| Experiment | Recipe File | Status |
|------------|-------------|--------|
| True Labels | baseline.yaml | ✅ |
| Random Labels | random_labels.yaml | ✅ |
| Shuffled Pixels | shuffled_pixels.yaml | ✅ |
| Random Pixels | random_pixels.yaml | ✅ |
| Gaussian Noise | gaussian_pixels.yaml | ✅ |

**Total: 5 experiments**

## Figure 1b/1c: Corruption Sweep

### Current Recipes
| Corruption % | Small Inception | Small AlexNet | MLP 1x512 |
|--------------|-----------------|---------------|-----------|
| 0% | baseline.yaml | alexnet_baseline.yaml | mlp_1x512_baseline.yaml |
| 10% | ✅ corruption_sweep/inception_corrupt_10.yaml | 📝 Create | 📝 Create |
| 20% | ✅ corruption_sweep/inception_corrupt_20.yaml | 📝 Create | 📝 Create |
| 30% | 📝 Create | 📝 Create | 📝 Create |
| 40% | 📝 Create | 📝 Create | 📝 Create |
| 50% | ✅ partial_corrupt_50.yaml | 📝 Create | 📝 Create |
| 60% | 📝 Create | 📝 Create | 📝 Create |
| 70% | 📝 Create | 📝 Create | 📝 Create |
| 80% | ✅ corruption_sweep/inception_corrupt_80.yaml | 📝 Create | 📝 Create |
| 90% | 📝 Create | 📝 Create | 📝 Create |
| 100% | random_labels.yaml | alexnet_random_labels.yaml | mlp_1x512_random_labels.yaml |

**Current: 4 recipes per model × 3 models = 12 total**
**Full sweep: 11 corruption levels × 3 models = 33 experiments**

## Appendix E / Table 4: Stress Tests (Random Labels + Regularization)

| Model | Random Labels + WD | Random Labels + Crop | Random Labels + Aug |
|-------|-------------------|---------------------|---------------------|
| **Small Inception** | ✅ stress_tests/inception_random_labels_wd.yaml | ✅ stress_tests/inception_random_labels_crop.yaml | ✅ stress_tests/inception_random_labels_aug.yaml |
| **Small AlexNet** | ✅ stress_tests/alexnet_random_labels_wd.yaml | - | - |
| **MLP 3x512** | ✅ stress_tests/mlp_3x512_random_labels_wd.yaml | - | - |
| **MLP 1x512** | ✅ stress_tests/mlp_1x512_random_labels_wd.yaml | - | - |

**Total: 6 experiments**

## Summary by Model

### Small Inception (Most Complete)
- Table 1: 5 configs ✅
- Figure 1a: 5 randomization types ✅
- Corruption sweep: 4 levels ✅ (expandable to 11)
- Stress tests: 3 configs ✅
- **Total: 17+ experiments**

### Small Inception No BN
- Table 1: 3 configs ✅
- **Total: 3 experiments**

### Small AlexNet
- Table 1: 5 configs ✅
- Corruption sweep: 0 levels (expandable to 11)
- Stress tests: 1 config ✅
- **Total: 6+ experiments**

### MLP 3x512
- Table 1: 3 configs ✅
- Stress tests: 1 config ✅
- **Total: 4 experiments**

### MLP 1x512
- Table 1: 3 configs ✅
- Corruption sweep: 0 levels (expandable to 11)
- Stress tests: 1 config ✅
- **Total: 4+ experiments**

## Grand Total
- **Currently Ready**: 34 experiments
- **With Full Corruption Sweep**: 60+ experiments

## Running Strategy

### Phase 1: Core Results (Table 1)
```bash
./run_all_experiments.sh --table1
```
**19 experiments** - Most important for paper

### Phase 2: Learning Curves (Figure 1a)
```bash
./run_all_experiments.sh --figure1a
```
**5 experiments** - Already covered by Table 1 baseline + random_labels

### Phase 3: Stress Tests (Appendix E)
```bash
./run_all_experiments.sh --stress-tests
```
**6 experiments** - Tests regularization effectiveness

### Phase 4: Corruption Sweep (Figure 1b/1c)
```bash
./run_all_experiments.sh --corruption-sweep
```
**4+ experiments** - Expandable to full sweep

### All at Once
```bash
./run_all_experiments.sh --all
```
**34+ experiments**

## W&B Organization

All experiments log to:
- **Project**: `FOLI-Project`
- **Entity**: `alirezasakhaeirad`
- **URL**: https://wandb.ai/alirezasakhaeirad/FOLI-Project

Recommended W&B tags/grouping:
- Group by: `model_name`
- Tag by: `experiment_type` (table1, figure1a, corruption_sweep, stress_test)
- Color by: `randomization` or `regularization`
