#!/bin/bash
# Script to generate and run all Table 1 experiments from paper 1611.03530v2
# This script creates all 19 experiment configurations exactly as reported in the paper

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in dry-run mode
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}Running in DRY-RUN mode - will only show commands${NC}\n"
fi

# Function to run or display experiment
run_experiment() {
    local exp_name=$1
    local model=$2
    local random_crop=$3
    local weight_decay=$4
    local randomization=$5
    local expected_train=$6
    local expected_test=$7
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Experiment: ${exp_name}${NC}"
    echo -e "  Model: ${model}"
    echo -e "  Random Crop: ${random_crop}"
    echo -e "  Weight Decay: ${weight_decay}"
    echo -e "  Randomization: ${randomization}"
    echo -e "  Expected Train Acc: ${expected_train}%"
    echo -e "  Expected Test Acc: ${expected_test}%"
    
    # Build command
    cmd="python train.py --model ${model} --dataset cifar10 --num_epochs 100 --use_wandb"
    
    if [[ "$random_crop" == "yes" ]]; then
        cmd="${cmd} --random_crop"
    fi
    
    if [[ "$weight_decay" != "no" ]]; then
        cmd="${cmd} --weight_decay ${weight_decay}"
    fi
    
    if [[ "$randomization" != "none" ]]; then
        cmd="${cmd} --randomization ${randomization}"
    fi
    
    echo -e "${GREEN}Command:${NC}"
    echo "  ${cmd}"
    echo ""
    
    if [[ "$DRY_RUN" == false ]]; then
        eval $cmd
        echo -e "${GREEN}✓ Completed: ${exp_name}${NC}\n"
    fi
}

echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Table 1: CIFAR-10 Experiments (Paper 1611.03530v2)${NC}"
echo -e "${YELLOW}  Total: 19 experiments across 5 model architectures${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}\n"

# ============================================================================
# INCEPTION (1,649,402 params) - 5 experiments
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Small Inception (1,649,402 params) - 5 experiments             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}\n"

# Exp 1: Inception + random crop + weight decay
run_experiment \
    "Inception: Random Crop + Weight Decay" \
    "small_inception" \
    "yes" \
    "0.0005" \
    "none" \
    "100.0" \
    "89.05"

# Exp 2: Inception + random crop + no weight decay
run_experiment \
    "Inception: Random Crop Only" \
    "small_inception" \
    "yes" \
    "no" \
    "none" \
    "100.0" \
    "89.31"

# Exp 3: Inception + no random crop + weight decay
run_experiment \
    "Inception: Weight Decay Only" \
    "small_inception" \
    "no" \
    "0.0005" \
    "none" \
    "100.0" \
    "86.03"

# Exp 4: Inception + no random crop + no weight decay (baseline)
run_experiment \
    "Inception: Baseline (No Regularization)" \
    "small_inception" \
    "no" \
    "no" \
    "none" \
    "100.0" \
    "85.75"

# Exp 5: Inception + random labels + no regularization
run_experiment \
    "Inception: Random Labels" \
    "small_inception" \
    "no" \
    "no" \
    "random_labels" \
    "100.0" \
    "9.78"

# ============================================================================
# INCEPTION w/o BatchNorm (1,649,402 params) - 3 experiments
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Inception w/o BatchNorm (1,649,402 params) - 3 experiments     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}\n"

# Exp 6: Inception No BN + no random crop + weight decay
run_experiment \
    "Inception No BN: Weight Decay Only" \
    "small_inception_no_bn" \
    "no" \
    "0.0005" \
    "none" \
    "100.0" \
    "83.00"

# Exp 7: Inception No BN + no random crop + no weight decay
run_experiment \
    "Inception No BN: Baseline (No Regularization)" \
    "small_inception_no_bn" \
    "no" \
    "no" \
    "none" \
    "100.0" \
    "82.00"

# Exp 8: Inception No BN + random labels + no regularization
run_experiment \
    "Inception No BN: Random Labels" \
    "small_inception_no_bn" \
    "no" \
    "no" \
    "random_labels" \
    "100.0" \
    "10.12"

# ============================================================================
# ALEXNET (1,387,786 params) - 5 experiments
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Small AlexNet (1,387,786 params) - 5 experiments               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}\n"

# Exp 9: AlexNet + random crop + weight decay
run_experiment \
    "AlexNet: Random Crop + Weight Decay" \
    "small_alexnet" \
    "yes" \
    "0.0005" \
    "none" \
    "99.90" \
    "81.22"

# Exp 10: AlexNet + random crop + no weight decay
run_experiment \
    "AlexNet: Random Crop Only" \
    "small_alexnet" \
    "yes" \
    "no" \
    "none" \
    "99.82" \
    "79.66"

# Exp 11: AlexNet + no random crop + weight decay
run_experiment \
    "AlexNet: Weight Decay Only" \
    "small_alexnet" \
    "no" \
    "0.0005" \
    "none" \
    "100.0" \
    "77.36"

# Exp 12: AlexNet + no random crop + no weight decay (baseline)
run_experiment \
    "AlexNet: Baseline (No Regularization)" \
    "small_alexnet" \
    "no" \
    "no" \
    "none" \
    "100.0" \
    "76.07"

# Exp 13: AlexNet + random labels + no regularization
run_experiment \
    "AlexNet: Random Labels" \
    "small_alexnet" \
    "no" \
    "no" \
    "random_labels" \
    "99.82" \
    "9.86"

# ============================================================================
# MLP 3×512 (1,735,178 params) - 3 experiments
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  MLP 3×512 (1,735,178 params) - 3 experiments                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}\n"

# Exp 14: MLP 3x512 + no random crop + weight decay
run_experiment \
    "MLP 3×512: Weight Decay Only" \
    "mlp_3x512" \
    "no" \
    "0.0001" \
    "none" \
    "100.0" \
    "53.35"

# Exp 15: MLP 3x512 + no random crop + no weight decay (baseline)
run_experiment \
    "MLP 3×512: Baseline (No Regularization)" \
    "mlp_3x512" \
    "no" \
    "no" \
    "none" \
    "100.0" \
    "52.39"

# Exp 16: MLP 3x512 + random labels + no regularization
run_experiment \
    "MLP 3×512: Random Labels" \
    "mlp_3x512" \
    "no" \
    "no" \
    "random_labels" \
    "100.0" \
    "10.48"

# ============================================================================
# MLP 1×512 (1,209,866 params) - 3 experiments
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  MLP 1×512 (1,209,866 params) - 3 experiments                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}\n"

# Exp 17: MLP 1x512 + no random crop + weight decay
run_experiment \
    "MLP 1×512: Weight Decay Only" \
    "mlp_1x512" \
    "no" \
    "0.0001" \
    "none" \
    "99.80" \
    "50.39"

# Exp 18: MLP 1x512 + no random crop + no weight decay (baseline)
run_experiment \
    "MLP 1×512: Baseline (No Regularization)" \
    "mlp_1x512" \
    "no" \
    "no" \
    "none" \
    "100.0" \
    "50.51"

# Exp 19: MLP 1x512 + random labels + no regularization
run_experiment \
    "MLP 1×512: Random Labels" \
    "mlp_1x512" \
    "no" \
    "no" \
    "random_labels" \
    "99.34" \
    "10.61"

# ============================================================================
# Summary
# ============================================================================
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    ALL EXPERIMENTS COMPLETE!                     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${YELLOW}Summary:${NC}"
echo -e "  • Small Inception: 5 experiments"
echo -e "  • Inception w/o BN: 3 experiments"
echo -e "  • Small AlexNet: 5 experiments"
echo -e "  • MLP 3×512: 3 experiments"
echo -e "  • MLP 1×512: 3 experiments"
echo -e "  ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}Total: 19 experiments${NC}\n"

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}To actually run the experiments, execute:${NC}"
    echo -e "  ${GREEN}./generate_table1_experiments.sh${NC}\n"
fi
