#!/bin/bash
# Batch job runner for RunAI
# This script is used as the entrypoint for non-interactive batch jobs

set -e

echo "============================================"
echo "Zhang et al. 2017 - Batch Job Runner"
echo "============================================"
echo "Started at: $(date)"
echo ""

# Check if config file is provided via environment variable
if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: EXPERIMENT_CONFIG environment variable not set"
    echo "Usage: Set EXPERIMENT_CONFIG=/workspace/recipes/baseline/inception_baseline.yaml"
    exit 1
fi

if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Config file not found: $EXPERIMENT_CONFIG"
    exit 1
fi

echo "Running experiment: $EXPERIMENT_CONFIG"
echo ""

# Run the experiment
cd /workspace
./run_experiment.sh --config "$EXPERIMENT_CONFIG" ${EXTRA_FLAGS}

echo ""
echo "============================================"
echo "Batch job completed at: $(date)"
echo "============================================"

