#!/bin/bash
# Master script to run all CIFAR10 experiments from Zhang et al. 2017
# Usage: ./recipes/run_all_recipes.sh [--use_wandb] [--category CATEGORY]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
USE_WANDB=""
FORCE_FLAG=""
CATEGORY="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --use_wandb)
            USE_WANDB="--use_wandb"
            shift
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--use_wandb] [--force] [--category baseline|randomization|regularization|ablation|all]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Zhang et al. 2017 - Running All Experiments${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

if [ -n "$USE_WANDB" ]; then
    echo -e "${GREEN}✓ W&B logging enabled${NC}"
    if [ -z "$WANDB_API_KEY" ]; then
        echo -e "${RED}ERROR: WANDB_API_KEY not set${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ W&B logging disabled${NC}"
fi

echo -e "Category: ${GREEN}${CATEGORY}${NC}"
echo ""

# Function to run experiments in a category
run_category() {
    local category=$1
    local category_path="recipes/${category}"
    
    if [ ! -d "$category_path" ]; then
        echo -e "${RED}Category not found: ${category}${NC}"
        return 1
    fi
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Running ${category} experiments${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    local count=0
    local total=$(ls -1 ${category_path}/*.yaml 2>/dev/null | wc -l)
    
    for recipe in ${category_path}/*.yaml; do
        if [ -f "$recipe" ]; then
            count=$((count + 1))
            local recipe_name=$(basename "$recipe" .yaml)
            
            echo -e "${GREEN}[${count}/${total}] Running: ${recipe_name}${NC}"
            echo -e "${YELLOW}Config: ${recipe}${NC}"
            echo ""
            
            # Run the experiment
            if ./run_experiment.sh --config "$recipe" $USE_WANDB $FORCE_FLAG; then
                echo -e "${GREEN}✓ Completed: ${recipe_name}${NC}"
            else
                echo -e "${RED}✗ Failed: ${recipe_name}${NC}"
                echo -e "${YELLOW}Continuing with next experiment...${NC}"
            fi
            
            echo ""
            echo -e "${BLUE}────────────────────────────────────────${NC}"
            echo ""
        fi
    done
    
    echo -e "${GREEN}Completed ${count} experiments in ${category}${NC}"
    echo ""
}

# Run experiments based on category
case $CATEGORY in
    baseline)
        run_category "baseline"
        ;;
    randomization)
        run_category "randomization"
        ;;
    regularization)
        run_category "regularization"
        ;;
    ablation)
        run_category "ablation"
        ;;
    all)
        run_category "baseline"
        run_category "randomization"
        run_category "regularization"
        run_category "ablation"
        ;;
    *)
        echo -e "${RED}Unknown category: ${CATEGORY}${NC}"
        echo "Valid categories: baseline, randomization, regularization, ablation, all"
        exit 1
        ;;
esac

echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Results saved in: ./results/"
echo "Checkpoints saved in: ./checkpoints/"

if [ -n "$USE_WANDB" ]; then
    echo "W&B Dashboard: https://wandb.ai/alirezasakhaeirad/FOLI-Project"
fi

