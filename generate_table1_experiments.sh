#!/bin/bash
# Script to generate and run all Table 1 experiments from paper 1611.03530v2
# This script uses the pre-generated recipes in recipes/table1/

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
    local recipe_file=$2
    local expected_train=$3
    local expected_test=$4
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Experiment: ${exp_name}${NC}"
    echo -e "  Recipe: recipes/table1/${recipe_file}"
    echo -e "  Expected Train Acc: ${expected_train}%"
    echo -e "  Expected Test Acc: ${expected_test}%"
    
    # Build command
    cmd="python train.py --config recipes/table1/${recipe_file}"
    
    echo -e "${GREEN}Command:${NC}"
    echo "  ${cmd}"
    echo ""
    
    if [[ "$DRY_RUN" == false ]]; then
        $cmd
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
echo -e "${BLUE}║  Inception (1,649,402 params) - 5 experiments                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}\n"

# Exp 1: Inception + random crop + weight decay
run_experiment \
    "Inception: Random Crop + Weight Decay" \
    "inception_crop_yes_wd_yes.yaml" \
    "100.0" \
    "89.05"

# Exp 2: Inception + random crop + no weight decay
run_experiment \
    "Inception: Random Crop Only" \
    "inception_crop_yes_wd_no.yaml" \
    "100.0" \
    "89.31"

# Exp 3: Inception + no random crop + weight decay
run_experiment \
    "Inception: Weight Decay Only" \
    "inception_crop_no_wd_yes.yaml" \
    "100.0" \
    "86.03"

# Exp 4: Inception + no random crop + no weight decay (baseline)
run_experiment \
    "Inception: Baseline (No Regularization)" \
    "inception_crop_no_wd_no.yaml" \
    "100.0" \
    "85.75"

# Exp 5: Inception + random labels
run_experiment \
    "Inception: Random Labels" \
    "inception_random_labels.yaml" \
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
    "inception_no_bn_crop_no_wd_yes.yaml" \
    "100.0" \
    "83.00"

# Exp 7: Inception No BN + no random crop + no weight decay
run_experiment \
    "Inception No BN: Baseline (No Regularization)" \
    "inception_no_bn_crop_no_wd_no.yaml" \
    "100.0" \
    "82.00"

# Exp 8: Inception No BN + random labels
run_experiment \
    "Inception No BN: Random Labels" \
    "inception_no_bn_random_labels.yaml" \
    "100.0" \
    "10.12"

# ============================================================================
# ALEXNET (1,387,786 params) - 5 experiments
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  AlexNet (1,387,786 params) - 5 experiments                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}\n"

# Exp 9: AlexNet + random crop + weight decay
run_experiment \
    "AlexNet: Random Crop + Weight Decay" \
    "small_alexnet_crop_yes_wd_yes.yaml" \
    "99.90" \
    "81.22"

# Exp 10: AlexNet + random crop + no weight decay
run_experiment \
    "AlexNet: Random Crop Only" \
    "small_alexnet_crop_yes_wd_no.yaml" \
    "99.82" \
    "79.66"

# Exp 11: AlexNet + no random crop + weight decay
run_experiment \
    "AlexNet: Weight Decay Only" \
    "small_alexnet_crop_no_wd_yes.yaml" \
    "100.0" \
    "77.36"

# Exp 12: AlexNet + no random crop + no weight decay (baseline)
run_experiment \
    "AlexNet: Baseline (No Regularization)" \
    "small_alexnet_crop_no_wd_no.yaml" \
    "100.0" \
    "76.07"

# Exp 13: AlexNet + random labels
run_experiment \
    "AlexNet: Random Labels" \
    "small_alexnet_random_labels.yaml" \
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
    "mlp_3x512_crop_no_wd_yes.yaml" \
    "100.0" \
    "53.35"

# Exp 15: MLP 3x512 + no random crop + no weight decay (baseline)
run_experiment \
    "MLP 3×512: Baseline (No Regularization)" \
    "mlp_3x512_crop_no_wd_no.yaml" \
    "100.0" \
    "52.39"

# Exp 16: MLP 3x512 + random labels
run_experiment \
    "MLP 3×512: Random Labels" \
    "mlp_3x512_random_labels.yaml" \
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
    "mlp_1x512_crop_no_wd_yes.yaml" \
    "99.80" \
    "50.39"

# Exp 18: MLP 1x512 + no random crop + no weight decay (baseline)
run_experiment \
    "MLP 1×512: Baseline (No Regularization)" \
    "mlp_1x512_crop_no_wd_no.yaml" \
    "100.0" \
    "50.51"

# Exp 19: MLP 1x512 + random labels
run_experiment \
    "MLP 1×512: Random Labels" \
    "mlp_1x512_random_labels.yaml" \
    "99.34" \
    "10.61"

# ============================================================================
# Summary
# ============================================================================
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    ALL EXPERIMENTS COMPLETE!                     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${YELLOW}Summary:${NC}"
echo -e "  • Inception: 5 experiments"
echo -e "  • Inception w/o BN: 3 experiments"
echo -e "  • AlexNet: 5 experiments"
echo -e "  • MLP 3×512: 3 experiments"
echo -e "  • MLP 1×512: 3 experiments"
echo -e "  ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}Total: 19 experiments${NC}\n"

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}To actually run the experiments, execute:${NC}"
    echo -e "  ${GREEN}./generate_table1_experiments.sh${NC}\n"
fi
