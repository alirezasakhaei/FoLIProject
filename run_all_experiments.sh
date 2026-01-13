#!/bin/bash
# Master script to run all experiments for the paper
# Usage: ./run_all_experiments.sh [--table1|--figure1a|--stress-tests|--all]

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a recipe
run_recipe() {
    local recipe=$1
    echo -e "${BLUE}Running experiment: ${recipe}${NC}"
    ./run_experiment.sh --config recipes/${recipe}.yaml
    echo -e "${GREEN}Completed: ${recipe}${NC}"
    echo ""
}

# Table 1 experiments
run_table1() {
    echo -e "${YELLOW}=== Running Table 1 Experiments ===${NC}"
    
    echo -e "${YELLOW}Small Inception (5 configs)${NC}"
    run_recipe "baseline"
    run_recipe "inception_weight_decay"
    run_recipe "inception_random_crop"
    run_recipe "regularized"
    run_recipe "inception_random_labels"
    
    echo -e "${YELLOW}Small Inception No BN (3 configs)${NC}"
    run_recipe "inception_no_bn_baseline"
    run_recipe "inception_no_bn_weight_decay"
    run_recipe "inception_no_bn_random_labels"
    
    echo -e "${YELLOW}Small AlexNet (5 configs)${NC}"
    run_recipe "alexnet_baseline"
    run_recipe "alexnet_weight_decay"
    run_recipe "alexnet_random_crop"
    run_recipe "alexnet_regularized"
    run_recipe "alexnet_random_labels"
    
    echo -e "${YELLOW}MLP 3x512 (3 configs)${NC}"
    run_recipe "mlp_3x512_baseline"
    run_recipe "mlp_3x512_weight_decay"
    run_recipe "mlp_3x512_random_labels"
    
    echo -e "${YELLOW}MLP 1x512 (3 configs)${NC}"
    run_recipe "mlp_1x512_baseline"
    run_recipe "mlp_1x512_weight_decay"
    run_recipe "mlp_1x512_random_labels"
    
    echo -e "${GREEN}=== Table 1 Experiments Complete (19 total) ===${NC}"
}

# Figure 1a experiments
run_figure1a() {
    echo -e "${YELLOW}=== Running Figure 1a Experiments ===${NC}"
    
    run_recipe "baseline"
    run_recipe "random_labels"
    run_recipe "shuffled_pixels"
    run_recipe "random_pixels"
    run_recipe "gaussian_pixels"
    
    echo -e "${GREEN}=== Figure 1a Experiments Complete (5 total) ===${NC}"
}

# Stress test experiments
run_stress_tests() {
    echo -e "${YELLOW}=== Running Stress Test Experiments (Appendix E) ===${NC}"
    
    for recipe in stress_tests/*.yaml; do
        recipe_name=$(basename ${recipe})
        echo -e "${BLUE}Running stress test: ${recipe_name}${NC}"
        ./run_experiment.sh --config ${recipe}
        echo -e "${GREEN}Completed: ${recipe_name}${NC}"
        echo ""
    done
    
    echo -e "${GREEN}=== Stress Test Experiments Complete ===${NC}"
}

# Corruption sweep experiments
run_corruption_sweep() {
    echo -e "${YELLOW}=== Running Corruption Sweep Experiments (Figure 1b/1c) ===${NC}"
    
    # Run existing corruption sweep configs
    for recipe in corruption_sweep/*.yaml; do
        if [ -f "$recipe" ]; then
            recipe_name=$(basename ${recipe})
            echo -e "${BLUE}Running corruption sweep: ${recipe_name}${NC}"
            ./run_experiment.sh --config ${recipe}
            echo -e "${GREEN}Completed: ${recipe_name}${NC}"
            echo ""
        fi
    done
    
    # Also run the 50% corruption from main directory
    run_recipe "partial_corrupt_50"
    
    echo -e "${GREEN}=== Corruption Sweep Experiments Complete ===${NC}"
}

# Main script logic
case "${1:-}" in
    --table1)
        run_table1
        ;;
    --figure1a)
        run_figure1a
        ;;
    --stress-tests)
        run_stress_tests
        ;;
    --corruption-sweep)
        run_corruption_sweep
        ;;
    --all)
        echo -e "${YELLOW}=== Running ALL Experiments ===${NC}"
        run_table1
        run_figure1a
        run_stress_tests
        run_corruption_sweep
        echo -e "${GREEN}=== ALL Experiments Complete! ===${NC}"
        ;;
    *)
        echo "Usage: $0 [--table1|--figure1a|--stress-tests|--corruption-sweep|--all]"
        echo ""
        echo "Options:"
        echo "  --table1            Run all Table 1 experiments (19 configs)"
        echo "  --figure1a          Run all Figure 1a experiments (5 configs)"
        echo "  --stress-tests      Run all stress test experiments (Appendix E)"
        echo "  --corruption-sweep  Run corruption sweep experiments (Figure 1b/1c)"
        echo "  --all               Run all experiments"
        echo ""
        echo "Examples:"
        echo "  $0 --table1         # Run only Table 1 experiments"
        echo "  $0 --all            # Run everything"
        exit 1
        ;;
esac
