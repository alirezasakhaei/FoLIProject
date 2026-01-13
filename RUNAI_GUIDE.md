# RunAI Usage Guide

Complete guide for running Zhang et al. 2017 experiments on RunAI cluster.

## 🐳 Docker Image

**Image:** `ic-registry.epfl.ch/folip/folip:latest`

The Docker image supports both:
- ✅ **Interactive sessions** - for development and debugging
- ✅ **Batch jobs** - for running experiments

## 🚀 Usage Modes

### Mode 1: Interactive Session (Recommended for Development)

Use this for exploring, debugging, and running experiments manually.

#### Start Interactive Session

```bash
python3 csub.py -n folip -g 1 --node-type a100-40g \
  -i ic-registry.epfl.ch/folip/folip:latest
```

**Note:** The command in csub.py already includes the proper override:
```bash
-- /bin/zsh -c 'source ~/.zshrc && sleep 43200'
```

This overrides the Docker CMD and gives you a shell.

#### Connect to Running Session

```bash
# Check status
runai describe job folip

# Connect to shell
runai exec folip -it -- zsh

# Or bash if zsh not available
runai exec folip -it -- bash
```

#### Inside the Container

```bash
# Navigate to workspace
cd /workspace

# Activate venv if needed (check if venv exists)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Validate recipes
python recipes/validate_recipes.py

# Run single experiment
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml

# Run with W&B (set API key first)
export WANDB_API_KEY=your_key_here
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --use_wandb

# Run category
./recipes/run_all_recipes.sh --category baseline --use_wandb

# Run all experiments
./recipes/run_all_recipes.sh --category all --use_wandb
```

### Mode 2: Batch Job (For Unattended Runs)

Use this for running experiments without manual intervention.

#### Submit Batch Job

```bash
# For a single experiment
python3 csub.py -n folip-baseline -g 1 --node-type a100-40g \
  -i ic-registry.epfl.ch/folip/folip:latest \
  -e EXPERIMENT_CONFIG=/workspace/recipes/baseline/inception_baseline.yaml \
  -e WANDB_API_KEY=your_key_here \
  -e EXTRA_FLAGS="--use_wandb" \
  -- /workspace/run_batch_job.sh

# For a category of experiments
python3 csub.py -n folip-all-baseline -g 1 --node-type a100-40g \
  -i ic-registry.epfl.ch/folip/folip:latest \
  -e WANDB_API_KEY=your_key_here \
  -e CATEGORY=baseline \
  -- /bin/bash -c "cd /workspace && ./recipes/run_all_recipes.sh --category baseline --use_wandb"
```

#### Monitor Batch Job

```bash
# Check status
runai describe job folip-baseline

# View logs (live)
runai logs folip-baseline -f

# View logs (all)
runai logs folip-baseline
```

## 🔧 Common Commands

### Job Management

```bash
# List all your jobs
runai list jobs

# Describe specific job
runai describe job <job-name>

# View logs
runai logs <job-name> -f

# Connect to running job
runai exec <job-name> -it -- bash

# Delete job
runai delete job <job-name>
```

### Inside Container

```bash
# Check GPU
nvidia-smi

# Check Python environment
python --version
pip list

# Check workspace
ls -la /workspace

# Check data
ls -la /workspace/data

# Check checkpoints
ls -la /workspace/checkpoints

# Check results
ls -la /workspace/results

# Validate recipes
python recipes/validate_recipes.py

# Run experiment
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml
```

## 📊 Experiment Workflows

### Workflow 1: Quick Test

```bash
# 1. Start interactive session
python3 csub.py -n folip-test -g 1 --node-type a100-40g \
  -i ic-registry.epfl.ch/folip/folip:latest

# 2. Connect
runai exec folip-test -it -- bash

# 3. Inside container - run quick test
cd /workspace
python recipes/validate_recipes.py
./run_experiment.sh --config recipes/baseline/mlp_1x512_baseline.yaml
# (MLP is faster than CNNs for testing)

# 4. Check results
ls -la results/
cat results/*.json | jq '.metrics.best_test_acc'

# 5. Clean up
exit
runai delete job folip-test
```

### Workflow 2: Run Single Experiment with W&B

```bash
# 1. Start interactive session
python3 csub.py -n folip-exp -g 1 --node-type a100-40g \
  -i ic-registry.epfl.ch/folip/folip:latest

# 2. Connect
runai exec folip-exp -it -- bash

# 3. Inside container
cd /workspace
export WANDB_API_KEY=your_key_here

# 4. Run experiment
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --use_wandb

# 5. Monitor on W&B
# https://wandb.ai/alirezasakhaeirad/FOLI-Project

# 6. When done
exit
runai delete job folip-exp
```

### Workflow 3: Run All Baseline Experiments

```bash
# 1. Start long-running interactive session
python3 csub.py -n folip-baseline -g 1 --node-type a100-40g \
  -i ic-registry.epfl.ch/folip/folip:latest

# 2. Connect
runai exec folip-baseline -it -- bash

# 3. Inside container - run in background with nohup
cd /workspace
export WANDB_API_KEY=your_key_here

nohup ./recipes/run_all_recipes.sh --category baseline --use_wandb > baseline.log 2>&1 &

# 4. Detach (Ctrl+D or exit)
exit

# 5. Check progress later
runai exec folip-baseline -it -- bash
cd /workspace
tail -f baseline.log

# 6. Or check individual results
ls -la results/
```

### Workflow 4: Run All Experiments (Full Reproduction)

```bash
# 1. Start long-running session (12 hours)
python3 csub.py -n folip-all -g 1 --node-type a100-40g \
  -i ic-registry.epfl.ch/folip/folip:latest

# 2. Connect
runai exec folip-all -it -- bash

# 3. Inside container - run all experiments
cd /workspace
export WANDB_API_KEY=your_key_here

# Run in background with nohup
nohup ./recipes/run_all_recipes.sh --category all --use_wandb > all_experiments.log 2>&1 &

# 4. Detach
exit

# 5. Check progress periodically
runai exec folip-all -it -- bash
cd /workspace
tail -f all_experiments.log

# Or check W&B dashboard
# https://wandb.ai/alirezasakhaeirad/FOLI-Project
```

## 🔍 Debugging

### Issue: Container Exits Immediately

**Problem:** Job status shows "Failed" or "ERROR"

**Solution:** Check logs
```bash
runai logs <job-name>
```

Common causes:
- Script has wrong line endings → Fixed in Dockerfile with dos2unix
- Missing execute permissions → Fixed in Dockerfile with chmod
- Wrong entrypoint → Use interactive mode with shell override

### Issue: Can't Connect to Job

**Problem:** `runai exec` fails

**Solution:** 
```bash
# Check if job is running
runai describe job <job-name>

# If status is not "Running", job may have crashed
runai logs <job-name>

# Delete and restart
runai delete job <job-name>
python3 csub.py -n <job-name> -g 1 --node-type a100-40g \
  -i ic-registry.epfl.ch/folip/folip:latest
```

### Issue: Experiment Already Completed

**Problem:** Script says "EXPERIMENT ALREADY COMPLETED"

**Solution:** Use `--force` flag
```bash
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --force
```

Or delete the results file:
```bash
rm results/small_inception_cifar10_baseline_*.json
```

### Issue: Out of Memory

**Problem:** CUDA out of memory error

**Solution:** Reduce batch size in recipe
```yaml
batch_size: 64  # or 32
```

### Issue: Data Not Found

**Problem:** "CIFAR10 data not found"

**Solution:** Download data inside container
```bash
cd /workspace
python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='./data', download=True)"
```

## 📦 Persistent Storage

### Using PVC (Persistent Volume Claim)

The csub.py script mounts `/nlpscratch` for persistent storage.

**Recommended structure:**
```bash
/nlpscratch/home/sakhaei/
├── folip/                    # Your project
│   ├── checkpoints/          # Persistent checkpoints
│   ├── results/              # Persistent results
│   └── data/                 # Downloaded datasets
```

**Inside container:**
```bash
# Create symlinks to persistent storage
mkdir -p /nlpscratch/home/sakhaei/folip/{checkpoints,results,data}

cd /workspace
ln -sf /nlpscratch/home/sakhaei/folip/checkpoints ./checkpoints
ln -sf /nlpscratch/home/sakhaei/folip/results ./results
ln -sf /nlpscratch/home/sakhaei/folip/data ./data

# Now checkpoints and results persist across jobs
```

## 🎯 Best Practices

### 1. Use Interactive Sessions for Development

```bash
# Start session
python3 csub.py -n folip-dev -g 1 --node-type a100-40g \
  -i ic-registry.epfl.ch/folip/folip:latest

# Connect and develop
runai exec folip-dev -it -- bash
```

### 2. Use Persistent Storage

```bash
# Link to persistent storage
ln -sf /nlpscratch/home/sakhaei/folip/checkpoints /workspace/checkpoints
ln -sf /nlpscratch/home/sakhaei/folip/results /workspace/results
```

### 3. Run Long Jobs with nohup

```bash
# Inside container
nohup ./recipes/run_all_recipes.sh --category all --use_wandb > all.log 2>&1 &
exit

# Check later
runai exec folip -it -- bash
tail -f /workspace/all.log
```

### 4. Monitor with W&B

```bash
# Always use W&B for long runs
export WANDB_API_KEY=your_key_here
./run_experiment.sh --config <recipe> --use_wandb
```

### 5. Clean Up After Jobs

```bash
# Delete completed jobs
runai delete job <job-name>

# List all jobs
runai list jobs

# Delete multiple
for job in $(runai list jobs | grep folip | awk '{print $1}'); do
    runai delete job $job
done
```

## 📋 Quick Reference

### Start Interactive Session
```bash
python3 csub.py -n folip -g 1 --node-type a100-40g \
  -i ic-registry.epfl.ch/folip/folip:latest
```

### Connect to Session
```bash
runai exec folip -it -- bash
```

### Run Single Experiment
```bash
cd /workspace
./run_experiment.sh --config recipes/baseline/inception_baseline.yaml --use_wandb
```

### Run Category
```bash
cd /workspace
./recipes/run_all_recipes.sh --category baseline --use_wandb
```

### Run All (Background)
```bash
cd /workspace
nohup ./recipes/run_all_recipes.sh --category all --use_wandb > all.log 2>&1 &
```

### Check Progress
```bash
tail -f /workspace/all.log
# Or check W&B: https://wandb.ai/alirezasakhaeirad/FOLI-Project
```

### Clean Up
```bash
runai delete job folip
```

## 🌐 Links

- **W&B Dashboard:** https://wandb.ai/alirezasakhaeirad/FOLI-Project
- **RunAI Docs:** https://docs.run.ai/
- **Project Docs:** See `COMPLETE_SETUP_SUMMARY.md`

---

**Happy Experimenting on RunAI! 🚀**

