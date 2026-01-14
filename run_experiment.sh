#!/bin/bash
set -e

# Script to run Zhang et al. 2017 experiments
# Usage: ./run_experiment.sh <config.yaml>
# Example: ./run_experiment.sh recipes/table1/inception_crop_no_wd_no.yaml

echo "============================================"
echo "Zhang et al. 2017 - Experiment Runner"
echo "============================================"

# Check for config argument
if [ $# -ne 1 ]; then
    echo "Error: Config file required"
    echo ""
    echo "Usage: ./run_experiment.sh <config.yaml>"
    echo ""
    echo "Examples:"
    echo "  ./run_experiment.sh recipes/table1/inception_crop_no_wd_no.yaml"
    echo "  ./run_experiment.sh recipes/table1/inception_crop_yes_wd_yes.yaml"
    exit 1
fi

CONFIG_FILE="$1"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check for WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY not set"
    echo ""
    echo "WandB logging is required. Please set your WandB API key:"
    echo "  export WANDB_API_KEY='your_key_here'"
    echo ""
    echo "Get your API key from: https://wandb.ai/authorize"
    exit 1
fi

echo "✓ WANDB_API_KEY found"
echo "Running experiment from config: $CONFIG_FILE"
echo ""

# Run experiment
python -m src.train.main "$CONFIG_FILE"

echo ""
echo "============================================"
echo "Experiment completed!"
echo "============================================"
