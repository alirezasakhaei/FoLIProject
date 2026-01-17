#!/bin/bash
set -e

# Script to run text classification experiments
# Usage: ./run_text_experiment.sh <config.yaml>
# Example: ./run_text_experiment.sh recipes/text/distilbert_imdb_true_labels.yaml

echo "============================================"
echo "Text Classification - Experiment Runner"
echo "============================================"

# Check for config argument
if [ $# -ne 1 ]; then
    echo "Error: Config file required"
    echo ""
    echo "Usage: ./run_text_experiment.sh <config.yaml>"
    echo ""
    echo "Examples:"
    echo "  ./run_text_experiment.sh recipes/text/distilbert_imdb_true_labels.yaml"
    echo "  ./run_text_experiment.sh recipes/text/distilbert_imdb_random_labels.yaml"
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
echo "Running text classification experiment from config: $CONFIG_FILE"
echo ""

# Run experiment
python -m src.train.text_main "$CONFIG_FILE"

echo ""
echo "============================================"
echo "Text experiment completed!"
echo "============================================"
