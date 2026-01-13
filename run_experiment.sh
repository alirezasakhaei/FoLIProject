#!/bin/bash
set -e

# Script to run Zhang et al. 2017 experiments
# Usage: ./run_experiment.sh --config recipes/baseline.yaml
#    or: ./run_experiment.sh --config recipes/baseline.yaml --use_wandb --force
#    or: ./run_experiment.sh --model small_inception --dataset cifar10 ...

echo "============================================"
echo "Zhang et al. 2017 - Experiment Runner"
echo "============================================"

# Check for WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. Weights & Biases logging will be disabled."
    USE_WANDB=""
else
    echo "✓ WANDB_API_KEY found"
    USE_WANDB="--use_wandb"
fi

# Parse arguments
CONFIG_FILE=""
FORCE_FLAG=""
DIRECT_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="--use_wandb"
            shift
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        *)
            DIRECT_ARGS+=("$1")
            shift
            ;;
    esac
done

# Run experiment
if [ -n "$CONFIG_FILE" ]; then
    echo "Running experiment from config: $CONFIG_FILE"
    python train.py --config "$CONFIG_FILE" $USE_WANDB $FORCE_FLAG
else
    echo "Running experiment with direct arguments"
    python train.py "${DIRECT_ARGS[@]}" $USE_WANDB $FORCE_FLAG
fi

echo "============================================"
echo "Experiment completed!"
echo "============================================"
