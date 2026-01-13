#!/usr/bin/env python3
"""
Validate all recipe YAML files for correctness.

Usage:
    python recipes/validate_recipes.py
"""

import os
import sys
from pathlib import Path
import yaml

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig


def validate_yaml_syntax(yaml_path):
    """Check if YAML file has valid syntax."""
    try:
        with open(yaml_path, 'r') as f:
            yaml.safe_load(f)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)


def validate_config(yaml_path):
    """Check if YAML can be loaded as ExperimentConfig."""
    try:
        config = ExperimentConfig.from_yaml(yaml_path)
        return True, None, config
    except Exception as e:
        return False, str(e), None


def validate_recipe(yaml_path):
    """Validate a single recipe file."""
    errors = []
    warnings = []
    
    # Check file exists
    if not os.path.exists(yaml_path):
        return False, [f"File not found: {yaml_path}"], []
    
    # Check YAML syntax
    valid_syntax, syntax_error = validate_yaml_syntax(yaml_path)
    if not valid_syntax:
        errors.append(f"Invalid YAML syntax: {syntax_error}")
        return False, errors, warnings
    
    # Check config loading
    valid_config, config_error, config = validate_config(yaml_path)
    if not valid_config:
        errors.append(f"Cannot load as ExperimentConfig: {config_error}")
        return False, errors, warnings
    
    # Validate config values
    if config.dataset != 'cifar10':
        warnings.append(f"Dataset is '{config.dataset}', expected 'cifar10'")
    
    if config.num_classes != 10:
        warnings.append(f"num_classes is {config.num_classes}, expected 10")
    
    if config.wandb_project != 'FOLI-Project':
        warnings.append(f"wandb_project is '{config.wandb_project}', expected 'FOLI-Project'")
    
    if config.wandb_entity != 'alirezasakhaeirad':
        warnings.append(f"wandb_entity is '{config.wandb_entity}', expected 'alirezasakhaeirad'")
    
    # Check randomization settings
    if 'random_labels' in str(yaml_path):
        if config.randomization != 'random_labels':
            warnings.append(f"Filename suggests random_labels but randomization is '{config.randomization}'")
    
    if 'baseline' in str(yaml_path):
        if config.randomization is not None:
            warnings.append(f"Baseline recipe has randomization set to '{config.randomization}'")
    
    # Check regularization settings for ablation studies
    if 'ablation' in str(yaml_path):
        if config.randomization != 'random_labels':
            warnings.append(f"Ablation study should use random_labels, but has '{config.randomization}'")
        
        if config.weight_decay == 0 and not config.random_crop and not config.augment_flip_rotate:
            warnings.append(f"Ablation study has no regularization enabled")
    
    return True, errors, warnings


def main():
    """Validate all recipe files."""
    recipes_dir = Path(__file__).parent
    
    print("=" * 60)
    print("Recipe Validation Report")
    print("=" * 60)
    print()
    
    categories = ['baseline', 'randomization', 'regularization', 'ablation']
    
    total_recipes = 0
    valid_recipes = 0
    invalid_recipes = 0
    total_warnings = 0
    
    for category in categories:
        category_path = recipes_dir / category
        
        if not category_path.exists():
            print(f"⚠️  Category not found: {category}")
            continue
        
        print(f"📁 {category.upper()}")
        print("-" * 60)
        
        yaml_files = sorted(category_path.glob('*.yaml'))
        
        if not yaml_files:
            print(f"  No recipes found in {category}/")
            print()
            continue
        
        for yaml_file in yaml_files:
            total_recipes += 1
            recipe_name = yaml_file.stem
            
            valid, errors, warnings = validate_recipe(yaml_file)
            
            if valid:
                valid_recipes += 1
                if warnings:
                    print(f"  ⚠️  {recipe_name}")
                    for warning in warnings:
                        print(f"      - {warning}")
                        total_warnings += 1
                else:
                    print(f"  ✅ {recipe_name}")
            else:
                invalid_recipes += 1
                print(f"  ❌ {recipe_name}")
                for error in errors:
                    print(f"      - {error}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total recipes:   {total_recipes}")
    print(f"Valid recipes:   {valid_recipes} ✅")
    print(f"Invalid recipes: {invalid_recipes} ❌")
    print(f"Total warnings:  {total_warnings} ⚠️")
    print()
    
    if invalid_recipes > 0:
        print("❌ Some recipes have errors. Please fix them before running experiments.")
        return 1
    elif total_warnings > 0:
        print("⚠️  All recipes are valid, but some have warnings.")
        return 0
    else:
        print("✅ All recipes are valid!")
        return 0


if __name__ == '__main__':
    sys.exit(main())

