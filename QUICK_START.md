# Quick Start Guide

## 🚀 Running Experiments

### Set W&B API Key (Required)
```bash
export WANDB_API_KEY=your_api_key
```

### Run All Table 1 Experiments (Recommended Start)
```bash
./run_all_experiments.sh --table1
```
This runs **19 experiments** covering all models with different regularization configs.

### Run Everything
```bash
./run_all_experiments.sh --all
```

### Run Individual Experiment
```bash
./run_experiment.sh --config recipes/baseline.yaml
```

## 📊 W&B Dashboard

Monitor your experiments at:
**https://wandb.ai/alirezasakhaeirad/FOLI-Project**

## 📁 Key Files

| File | Purpose |
|------|---------|
| `run_all_experiments.sh` | Master script to run experiments by category |
| `run_experiment.sh` | Run a single experiment |
| `recipes/README.md` | Complete recipe documentation |
| `EXPERIMENT_MATRIX.md` | Visual matrix of all experiments |
| `COMMANDS.md` | Detailed command reference |

## 🔬 Experiment Categories

### Table 1 (19 experiments)
All models × regularization configurations
```bash
./run_all_experiments.sh --table1
```

### Figure 1a (5 experiments)
Learning curves with different randomizations
```bash
./run_all_experiments.sh --figure1a
```

### Stress Tests (6 experiments)
Random labels + explicit regularization
```bash
./run_all_experiments.sh --stress-tests
```

### Corruption Sweep (4+ experiments)
Partial label corruption at different levels
```bash
./run_all_experiments.sh --corruption-sweep
```

## 🎯 Models Available

- `small_inception` - Small Inception with BatchNorm
- `small_inception_no_bn` - Small Inception without BatchNorm
- `small_alexnet` - Small AlexNet
- `mlp_3x512` - MLP with 3 hidden layers of 512 units
- `mlp_1x512` - MLP with 1 hidden layer of 512 units

## 🔧 Customization

To create a custom experiment:
1. Copy an existing recipe from `recipes/`
2. Modify the parameters (model, randomization, regularization)
3. Run with: `./run_experiment.sh --config recipes/your_recipe.yaml`

## 📈 Expected Runtime

- Single experiment: ~1-2 hours (100 epochs on CIFAR-10)
- Table 1 (19 experiments): ~20-40 hours
- All experiments (34+): ~40-70 hours

## ✅ Checklist

- [ ] Set WANDB_API_KEY
- [ ] Review `EXPERIMENT_MATRIX.md` to understand experiment structure
- [ ] Start with Table 1: `./run_all_experiments.sh --table1`
- [ ] Monitor progress on W&B dashboard
- [ ] Run additional experiments as needed

## 🆘 Troubleshooting

**Issue**: W&B not logging
**Solution**: Check `export WANDB_API_KEY=your_key` is set

**Issue**: CUDA out of memory
**Solution**: Reduce batch size in recipe YAML file

**Issue**: Checkpoint already exists
**Solution**: Experiments auto-resume from checkpoints. Delete checkpoint to restart.

## 📚 Documentation

- **Full recipe list**: `recipes/README.md`
- **Experiment matrix**: `EXPERIMENT_MATRIX.md`
- **Setup summary**: `EXPERIMENT_SETUP_SUMMARY.md`
- **Command reference**: `COMMANDS.md`
