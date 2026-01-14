# Checkpoint and Resumption System

This document explains the checkpoint, resumption, and completion detection system for the Zhang et al. 2017 experiments.

## 🎯 Key Features

### 1. **Automatic Checkpoint Saving**
- ✅ Checkpoints saved after every epoch
- ✅ Includes model, optimizer, scheduler, and all metrics
- ✅ Best model saved separately
- ✅ Unique naming based on experiment configuration

### 2. **Automatic Resumption**
- ✅ Automatically resumes from last checkpoint if interrupted
- ✅ Restores all training state (epoch, optimizer, metrics)
- ✅ Continues from exact point of interruption
- ✅ No data loss or duplicate training

### 3. **Completion Detection**
- ✅ Detects if experiment already completed
- ✅ Prevents accidental re-runs
- ✅ Shows previous results
- ✅ Can force re-run with `--force` flag

### 4. **Complete Results Saving**
- ✅ All metrics saved to JSON after completion
- ✅ Includes config, all epoch metrics, and summary
- ✅ Timestamped for tracking
- ✅ Easy to parse and analyze

## 📁 File Structure

### Checkpoints Directory (`./checkpoints/`)

```
checkpoints/
├── inception_cifar10_baseline_checkpoint.pth       # Latest checkpoint
├── inception_cifar10_baseline_best.pth             # Best model
├── inception_cifar10_random_labels_checkpoint.pth  # Another experiment
└── inception_cifar10_random_labels_best.pth
```

**Checkpoint Contents:**
- `epoch`: Current epoch number
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: Learning rate scheduler state (if used)
- `best_test_acc`: Best test accuracy so far
- `config`: Full experiment configuration
- `all_metrics`: All metrics from all epochs (for resumption)
- `experiment_id`: Unique experiment identifier

### Results Directory (`./results/`)

```
results/
├── inception_cifar10_baseline_20260113_143022.json
├── inception_cifar10_random_labels_20260113_150145.json
└── inception_cifar10_baseline_wd0.0005_crop_20260113_153210.json
```

**Results File Contents:**
```json
{
  "experiment_id": "inception_cifar10_baseline",
  "config": {
    "model_name": "inception",
    "dataset": "cifar10",
    "randomization": null,
    "weight_decay": 0.0,
    ...
  },
  "metrics": {
    "epochs": [1, 2, 3, ..., 100],
    "train_loss": [2.3, 1.8, 1.5, ...],
    "train_acc": [10.2, 35.6, 48.3, ...],
    "test_loss": [2.1, 1.7, 1.4, ...],
    "test_acc": [15.3, 40.2, 52.1, ...],
    "best_test_acc": 78.5,
    "final_train_acc": 99.2,
    "final_test_acc": 77.8
  },
  "timestamp": "20260113_143022",
  "completed": true,
  "total_epochs": 100,
  "target_epochs": 100
}
```

## 🔄 How It Works

### Experiment ID Generation

Each experiment gets a unique ID based on its configuration:

```
{model}_{dataset}_{randomization}_{regularizers}_{corruption}
```

Examples:
- `inception_cifar10_baseline`
- `inception_cifar10_random_labels`
- `inception_cifar10_baseline_wd0.0005_crop`
- `inception_cifar10_partial_corrupt_corrupt20`

### Workflow

#### 1. Starting a New Experiment

```bash
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml
```

**What happens:**
1. Generate experiment ID from config
2. Check if experiment already completed → No
3. Check for existing checkpoint → No
4. Start training from epoch 1
5. Save checkpoint after each epoch
6. Save final results when complete

#### 2. Resuming an Interrupted Experiment

```bash
# Experiment was interrupted at epoch 45
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml
```

**What happens:**
1. Generate experiment ID from config
2. Check if experiment already completed → No
3. Check for existing checkpoint → Yes (epoch 45)
4. Load checkpoint:
   - Restore model weights
   - Restore optimizer state
   - Restore all metrics from epochs 1-45
5. Resume training from epoch 46
6. Continue until epoch 100
7. Save final results with all 100 epochs

#### 3. Trying to Re-run Completed Experiment

```bash
# Experiment already completed
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml
```

**What happens:**
```
============================================
⚠️  EXPERIMENT ALREADY COMPLETED
============================================
Results file: ./results/inception_cifar10_baseline_20260113_143022.json

To re-run this experiment, either:
  1. Add --force flag: ./run_experiment.sh --config <recipe> --force
  2. Delete the results file: ./results/inception_cifar10_baseline_20260113_143022.json
  3. Delete the checkpoint: ./checkpoints/inception_cifar10_baseline_checkpoint.pth
============================================

Previous Results:
  Best test accuracy: 78.50%
  Final test accuracy: 77.80%
  Total epochs: 100
  Completed: 20260113_143022
```

**Experiment exits without re-running.**

#### 4. Force Re-running a Completed Experiment

```bash
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --force
```

**What happens:**
1. Detect experiment already completed
2. See `--force` flag → Proceed anyway
3. Start fresh from epoch 1 (or resume from checkpoint if exists)
4. Save new results file with new timestamp

## 🚀 Usage Examples

### Basic Usage

```bash
# Run experiment (will resume if interrupted)
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml

# Run with W&B logging
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --use_wandb

# Force re-run even if completed
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --force

# Force re-run with W&B
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --use_wandb --force
```

### Batch Execution

```bash
# Run all baseline experiments (skips completed ones)
./recipes/run_all_recipes.sh --category baseline

# Run all with W&B logging
./recipes/run_all_recipes.sh --category baseline --use_wandb

# Force re-run all (even completed ones)
./recipes/run_all_recipes.sh --category baseline --force

# Run everything with W&B and force
./recipes/run_all_recipes.sh --category all --use_wandb --force
```

## 🔍 Checking Experiment Status

### Method 1: Check Results Directory

```bash
ls -lh results/
```

If you see a results file for your experiment, it's completed.

### Method 2: Check Checkpoint Directory

```bash
ls -lh checkpoints/
```

If you see a checkpoint file, the experiment is in progress or completed.

### Method 3: Try Running

```bash
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml
```

If completed, you'll see the completion message and previous results.

## 🛠️ Managing Experiments

### Delete Checkpoint to Start Fresh

```bash
# Delete specific experiment checkpoint
rm checkpoints/inception_cifar10_baseline_checkpoint.pth
rm checkpoints/inception_cifar10_baseline_best.pth

# Delete all checkpoints
rm checkpoints/*.pth
```

### Delete Results to Allow Re-run

```bash
# Delete specific experiment results
rm results/inception_cifar10_baseline_*.json

# Delete all results
rm results/*.json
```

### Clean Everything

```bash
# Clean all checkpoints and results
rm -rf checkpoints/*.pth results/*.json
```

## 📊 Analyzing Results

### Load Results in Python

```python
import json

# Load results
with open('results/inception_cifar10_baseline_20260113_143022.json', 'r') as f:
    results = json.load(f)

# Access metrics
epochs = results['metrics']['epochs']
train_acc = results['metrics']['train_acc']
test_acc = results['metrics']['test_acc']

# Plot learning curves
import matplotlib.pyplot as plt
plt.plot(epochs, train_acc, label='Train')
plt.plot(epochs, test_acc, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

# Print summary
print(f"Best test accuracy: {results['metrics']['best_test_acc']:.2f}%")
print(f"Final test accuracy: {results['metrics']['final_test_acc']:.2f}%")
```

### Compare Multiple Experiments

```python
import json
import glob

# Load all results
results_files = glob.glob('results/*.json')
experiments = []

for filepath in results_files:
    with open(filepath, 'r') as f:
        experiments.append(json.load(f))

# Compare best test accuracies
for exp in experiments:
    exp_id = exp['experiment_id']
    best_acc = exp['metrics']['best_test_acc']
    print(f"{exp_id}: {best_acc:.2f}%")
```

## ⚠️ Important Notes

### 1. Checkpoint Naming

Checkpoints are named based on experiment configuration, not the recipe filename. This means:

```bash
# These two commands will use the SAME checkpoint
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml
./run_experiment.sh --model inception --dataset cifar10
```

Both generate experiment ID: `inception_cifar10_baseline`

### 2. Changing Configuration

If you modify the recipe configuration, it will create a NEW experiment:

```yaml
# Original: inception_baseline.yaml
weight_decay: 0.0  # Experiment ID: inception_cifar10_baseline

# Modified: inception_baseline.yaml
weight_decay: 0.0005  # NEW Experiment ID: inception_cifar10_baseline_wd0.0005
```

### 3. Multiple Results Files

You can have multiple results files for the same experiment (different runs):

```
results/
├── inception_cifar10_baseline_20260113_143022.json  # First run
├── inception_cifar10_baseline_20260114_091530.json  # Second run (with --force)
└── inception_cifar10_baseline_20260115_163245.json  # Third run
```

The system only checks if ANY completed results file exists.

### 4. Checkpoint vs Results

- **Checkpoint**: Saved after every epoch (for resumption)
- **Results**: Saved only when training completes

If you have a checkpoint but no results, the experiment is in progress or was interrupted.

## 🐛 Troubleshooting

### Problem: "Experiment already completed" but I want to re-run

**Solution:** Use `--force` flag
```bash
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --force
```

### Problem: Training not resuming from checkpoint

**Check:**
1. Is the checkpoint file present in `checkpoints/`?
2. Does the checkpoint name match the experiment ID?
3. Is the checkpoint corrupted? (Try deleting and starting fresh)

### Problem: Results file missing even though training completed

**Possible causes:**
1. Training was interrupted before saving results
2. Disk full during save
3. Permission issues

**Solution:** Re-run the experiment (it will resume from checkpoint and save results)

### Problem: Want to resume but start from different epoch

**Solution:** Manually edit the checkpoint file or delete it to start fresh

### Problem: Checkpoint taking too much disk space

**Solution:** Delete old checkpoints you don't need
```bash
# Keep only best models
rm checkpoints/*_checkpoint.pth
# (Keep *_best.pth files)
```

## 📈 Best Practices

1. **Let experiments complete naturally** - Don't interrupt unless necessary
2. **Use `--force` sparingly** - Only when you really want to re-run
3. **Back up results** - Copy `results/` directory periodically
4. **Monitor disk space** - Checkpoints can be large (100-500 MB each)
5. **Use W&B for tracking** - Better than relying only on local files
6. **Check completion before batch runs** - Use validation to avoid wasting compute
7. **Clean up old checkpoints** - Delete checkpoints after experiments complete

## 🎓 Summary

The checkpoint system provides:
- ✅ **Automatic resumption** from any interruption
- ✅ **Completion detection** to avoid duplicate work
- ✅ **Complete metrics** saved for analysis
- ✅ **Flexible control** with `--force` flag
- ✅ **Unique naming** based on configuration
- ✅ **Easy management** of experiments

You can safely run experiments knowing that:
- Interruptions won't lose progress
- Completed experiments won't re-run accidentally
- All metrics are preserved
- Results are easily accessible

---

**Happy Experimenting! 🚀**

