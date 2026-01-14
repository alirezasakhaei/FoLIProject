# Quick Start Guide - Experiment Recipes

Get started running Zhang et al. 2017 experiments in minutes!

## 🚀 Quick Start (5 minutes)

### 1. Validate All Recipes

```bash
python recipes/validate_recipes.py
```

Expected output: `✅ All recipes are valid!`

### 2. Run Your First Experiment

```bash
# Run baseline Inception (should achieve ~75-80% test accuracy)
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml
```

### 3. Run the Key Experiment (Random Labels)

```bash
# This demonstrates the paper's central finding
./run_experiment.sh --config recipes/randomization/inception_random_labels.yaml
```

**Expected Results:**
- Training accuracy: 100% (0% training error)
- Test accuracy: ~10% (random chance)

This proves that networks can perfectly memorize random labels!

## 📊 Enable W&B Logging

```bash
# Set your API key
export WANDB_API_KEY=your_key_here

# Run with W&B logging
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --use_wandb
```

View results at: https://wandb.ai/alirezasakhaeirad/FOLI-Project

## 🎯 Common Use Cases

### Run a Specific Category

```bash
# Baseline experiments (4 recipes, ~10-12 hours)
./recipes/run_all_recipes.sh --category baseline

# Randomization experiments (9 recipes, ~27-36 hours)
./recipes/run_all_recipes.sh --category randomization

# Regularization experiments (4 recipes, ~12-16 hours)
./recipes/run_all_recipes.sh --category regularization

# Ablation studies (6 recipes, ~18-24 hours)
./recipes/run_all_recipes.sh --category ablation
```

### Run Everything

```bash
# Run all 23 experiments (~80-100 GPU hours)
./recipes/run_all_recipes.sh --category all --use_wandb
```

### Create a Custom Recipe

```bash
# Copy an existing recipe
cp recipes/baseline/inception_baseline.yaml recipes/my_experiment.yaml

# Edit the parameters
vim recipes/my_experiment.yaml

# Run your custom experiment
./run_experiment.sh --config recipes/my_experiment.yaml
```

## 🔍 Understanding Recipe Structure

Every recipe is a YAML file with these key sections:

```yaml
# Model selection
model_name: inception  # or small_alexnet, mlp_1x512, mlp_3x512

# Dataset
dataset: cifar10
data_root: ./data

# Randomization (null for baseline)
randomization: null  # or random_labels, gaussian_pixels, etc.
corruption_prob: 0.0  # for partial_corrupt

# Regularization
weight_decay: 0.0      # L2 regularization
random_crop: false     # Data augmentation
augment_flip_rotate: false  # More augmentation

# Training
batch_size: 128
num_epochs: 100
learning_rate: 0.01
momentum: 0.9
optimizer: sgd

# W&B logging
use_wandb: false
wandb_project: FOLI-Project
wandb_entity: alirezasakhaeirad

# Reproducibility
seed: 42
device: cuda
```

## 📖 Recipe Categories Explained

### 1. **Baseline** (`baseline/`)
Standard training on true labels, no regularization.
- **Purpose:** Establish baseline performance
- **Expected:** Good generalization (70-80% test accuracy)

### 2. **Randomization** (`randomization/`)
Various forms of data randomization.
- **Purpose:** Test memorization capacity
- **Expected:** Perfect training accuracy, poor test accuracy

### 3. **Regularization** (`regularization/`)
Adding explicit regularization to baseline.
- **Purpose:** Test if regularization improves generalization
- **Expected:** Slightly better test accuracy

### 4. **Ablation** (`ablation/`)
Regularization on random labels.
- **Purpose:** Test if regularization prevents memorization
- **Expected:** Networks still fit random labels!

## 🎓 Reproducing Key Paper Findings

### Finding 1: Networks Fit Random Labels

```bash
# Run these two experiments
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml
./run_experiment.sh --config recipes/randomization/inception_random_labels.yaml

# Compare results:
# Baseline: High test accuracy (generalization)
# Random labels: Low test accuracy (no generalization)
# BUT both achieve high training accuracy!
```

### Finding 2: Regularization Doesn't Prevent Memorization

```bash
# Random labels WITHOUT regularization
./run_experiment.sh --config recipes/randomization/inception_random_labels.yaml

# Random labels WITH regularization
./run_experiment.sh --config recipes/ablation/inception_random_labels_weight_decay.yaml

# Both achieve 100% training accuracy!
```

### Finding 3: Networks Can Fit Pure Noise

```bash
# Replace images with Gaussian noise
./run_experiment.sh --config recipes/randomization/inception_gaussian_pixels.yaml

# Still achieves 100% training accuracy!
```

### Finding 4: Gradual Degradation with Noise

```bash
# Run partial corruption experiments
./run_experiment.sh --config recipes/randomization/inception_partial_corrupt_20.yaml
./run_experiment.sh --config recipes/randomization/inception_partial_corrupt_50.yaml
./run_experiment.sh --config recipes/randomization/inception_partial_corrupt_80.yaml

# Test accuracy degrades smoothly: ~70% → ~50% → ~25%
```

## 🛠️ Troubleshooting

### "CUDA out of memory"

Reduce batch size in the recipe:
```yaml
batch_size: 64  # or 32
```

### "WANDB_API_KEY not set"

Either set the key:
```bash
export WANDB_API_KEY=your_key_here
```

Or run without W&B:
```bash
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml
# (don't add --use_wandb flag)
```

### Experiment takes too long

Start with smaller models:
```bash
# MLPs are faster than CNNs
./run_experiment.sh --config recipes/baseline/mlp_1x512_baseline.yaml
```

Or reduce epochs in the recipe:
```yaml
num_epochs: 50  # instead of 100
```

## 📁 File Locations

After running experiments, you'll find:

```
Project/
├── checkpoints/          # Model checkpoints
│   ├── inception_cifar10_checkpoint.pth
│   └── inception_cifar10_best.pth
├── results/              # JSON result files
│   └── inception_cifar10_baseline_20260113_123456.json
└── recipes/              # All experiment configs
    ├── baseline/
    ├── randomization/
    ├── regularization/
    └── ablation/
```

## 🎯 Next Steps

1. **Start small:** Run 1-2 baseline experiments
2. **Verify setup:** Check that results match expected values
3. **Scale up:** Run category-by-category
4. **Analyze:** Compare results across experiments
5. **Customize:** Create your own recipes for new experiments

## 📚 Additional Resources

- **Full Index:** See `EXPERIMENTS_INDEX.md` for complete experiment list
- **Detailed Guide:** See `README.md` for in-depth documentation
- **Paper:** `1611.03530v2 (1).pdf` in project root
- **W&B Dashboard:** https://wandb.ai/alirezasakhaeirad/FOLI-Project

## 💡 Pro Tips

1. **Use W&B:** Makes comparing experiments much easier
2. **Start with Priority 1 experiments:** See EXPERIMENTS_INDEX.md
3. **Run overnight:** Full experiments take 3-4 hours each
4. **Check validation first:** Always run `validate_recipes.py` before batch runs
5. **Monitor GPU usage:** Use `nvidia-smi` to check utilization

---

**Happy Experimenting! 🚀**

Questions? Check the main README.md or the paper for details.

