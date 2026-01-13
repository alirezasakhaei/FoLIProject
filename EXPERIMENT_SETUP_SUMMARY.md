# Experiment Setup Summary

## Changes Made

### 1. Updated W&B Configuration in All YAML Files

All recipe files now use:
- **Project**: `FOLI-Project`
- **Entity**: `alirezasakhaeirad`

Updated files:
- ✅ baseline.yaml
- ✅ regularized.yaml
- ✅ random_labels.yaml
- ✅ alexnet_baseline.yaml
- ✅ mlp_3x512_baseline.yaml
- ✅ partial_corrupt_50.yaml
- ✅ shuffled_pixels.yaml
- ✅ random_pixels.yaml
- ✅ gaussian_pixels.yaml

### 2. Created Comprehensive Recipe Files

#### Table 1 Experiments (19 total)

**Small Inception (5 configs):**
- ✅ baseline.yaml (no regularization)
- ✅ inception_weight_decay.yaml (weight decay only)
- ✅ inception_random_crop.yaml (random crop only)
- ✅ regularized.yaml (weight decay + random crop)
- ✅ inception_random_labels.yaml (random labels)

**Small Inception No BN (3 configs):**
- ✅ inception_no_bn_baseline.yaml
- ✅ inception_no_bn_weight_decay.yaml
- ✅ inception_no_bn_random_labels.yaml

**Small AlexNet (5 configs):**
- ✅ alexnet_baseline.yaml
- ✅ alexnet_weight_decay.yaml
- ✅ alexnet_random_crop.yaml
- ✅ alexnet_regularized.yaml
- ✅ alexnet_random_labels.yaml

**MLP 3x512 (3 configs):**
- ✅ mlp_3x512_baseline.yaml
- ✅ mlp_3x512_weight_decay.yaml
- ✅ mlp_3x512_random_labels.yaml

**MLP 1x512 (3 configs):**
- ✅ mlp_1x512_baseline.yaml
- ✅ mlp_1x512_weight_decay.yaml
- ✅ mlp_1x512_random_labels.yaml

#### Figure 1a Experiments (5 total)
- ✅ baseline.yaml (true labels)
- ✅ random_labels.yaml
- ✅ shuffled_pixels.yaml
- ✅ random_pixels.yaml
- ✅ gaussian_pixels.yaml

#### Figure 1b/1c: Corruption Sweep
Created in `corruption_sweep/` subdirectory:
- ✅ inception_corrupt_10.yaml (10% corruption)
- ✅ inception_corrupt_20.yaml (20% corruption)
- ✅ inception_corrupt_80.yaml (80% corruption)
- ✅ partial_corrupt_50.yaml (50% corruption - main directory)

**Note:** Additional corruption levels (0.0, 0.3, 0.4, 0.6, 0.7, 0.9, 1.0) can be created by copying and modifying these templates.

#### Appendix E / Table 4: Stress Tests (6 total)
Created in `stress_tests/` subdirectory:
- ✅ inception_random_labels_wd.yaml
- ✅ inception_random_labels_crop.yaml
- ✅ inception_random_labels_aug.yaml
- ✅ alexnet_random_labels_wd.yaml
- ✅ mlp_3x512_random_labels_wd.yaml
- ✅ mlp_1x512_random_labels_wd.yaml

### 3. Created Master Execution Script

✅ **run_all_experiments.sh** - Master script to run experiments by category:
- `--table1` - Run all 19 Table 1 experiments
- `--figure1a` - Run all 5 Figure 1a experiments
- `--stress-tests` - Run all stress test experiments
- `--corruption-sweep` - Run corruption sweep experiments
- `--all` - Run everything

Features:
- Colored output for progress tracking
- Sequential execution with error handling
- Organized by experiment type

### 4. Documentation

✅ **recipes/README.md** - Comprehensive documentation including:
- Complete list of all experiments
- Recipe structure explanation
- Model names and configurations
- Randomization modes
- Weight decay values
- Instructions for running all experiments

✅ **COMMANDS.md** - Updated with:
- New recipe organization by experiment type
- Reference to recipes/README.md
- Master script usage instructions
- W&B configuration details

## Total Experiment Count

- **Table 1**: 19 experiments (all models × regularization configs)
- **Figure 1a**: 5 experiments (learning curves)
- **Figure 1b/1c**: 4+ experiments (corruption sweep - expandable)
- **Appendix E**: 6 experiments (stress tests)

**Total**: 34+ experiments ready to run

## How to Run All Experiments

### Option 1: Use Master Script (Recommended)
```bash
# Run all Table 1 experiments
./run_all_experiments.sh --table1

# Run everything
./run_all_experiments.sh --all
```

### Option 2: Run Individual Recipes
```bash
./run_experiment.sh --config recipes/baseline.yaml
./run_experiment.sh --config recipes/inception_weight_decay.yaml
# etc.
```

### Option 3: Loop Through Categories
```bash
# Table 1 - Small Inception
for recipe in baseline inception_weight_decay inception_random_crop regularized inception_random_labels; do
  ./run_experiment.sh --config recipes/${recipe}.yaml
done
```

## Next Steps

1. **Set W&B API Key**:
   ```bash
   export WANDB_API_KEY=your_api_key
   ```

2. **Start Running Experiments**:
   ```bash
   ./run_all_experiments.sh --table1
   ```

3. **Monitor Progress** on W&B:
   https://wandb.ai/alirezasakhaeirad/FOLI-Project

4. **Optional: Create Additional Corruption Sweep Configs**
   - Copy existing corruption sweep templates
   - Modify `corruption_prob` values (0.0, 0.3, 0.4, 0.6, 0.7, 0.9, 1.0)
   - For different models (AlexNet, MLP 1x512)

## Answer to Your Question

> "shouldn't I loop through all models for the paper?"

**Yes, absolutely!** And now you can! The recipes are organized so you can:

1. **Loop through all models for Table 1** (19 experiments covering all 5 models with different regularization configs)
2. **Loop through all randomization modes for Figure 1a** (5 experiments)
3. **Loop through corruption levels for Figure 1b/1c** (expandable to all corruption probabilities)
4. **Loop through stress tests for Appendix E** (6 experiments)

The master script `run_all_experiments.sh` does exactly this - it loops through all models and configurations systematically!

## Files Created/Modified

### Created:
- recipes/inception_weight_decay.yaml
- recipes/inception_random_crop.yaml
- recipes/inception_random_labels.yaml
- recipes/inception_no_bn_baseline.yaml
- recipes/inception_no_bn_weight_decay.yaml
- recipes/inception_no_bn_random_labels.yaml
- recipes/alexnet_weight_decay.yaml
- recipes/alexnet_random_crop.yaml
- recipes/alexnet_regularized.yaml
- recipes/alexnet_random_labels.yaml
- recipes/mlp_3x512_weight_decay.yaml
- recipes/mlp_3x512_random_labels.yaml
- recipes/mlp_1x512_baseline.yaml
- recipes/mlp_1x512_weight_decay.yaml
- recipes/mlp_1x512_random_labels.yaml
- recipes/corruption_sweep/inception_corrupt_10.yaml
- recipes/corruption_sweep/inception_corrupt_20.yaml
- recipes/corruption_sweep/inception_corrupt_80.yaml
- recipes/stress_tests/inception_random_labels_wd.yaml
- recipes/stress_tests/inception_random_labels_crop.yaml
- recipes/stress_tests/inception_random_labels_aug.yaml
- recipes/stress_tests/alexnet_random_labels_wd.yaml
- recipes/stress_tests/mlp_3x512_random_labels_wd.yaml
- recipes/stress_tests/mlp_1x512_random_labels_wd.yaml
- recipes/README.md
- run_all_experiments.sh

### Modified:
- All existing YAML files (updated W&B project/entity)
- COMMANDS.md (updated with new recipe organization)
