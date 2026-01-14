"""
Display functions for training progress and configuration.
"""
from datetime import datetime
from src.config import ExperimentConfig
from .utils import get_model_params_info


def get_experiment_id(config: ExperimentConfig) -> str:
    """Generate unique experiment identifier from config."""
    model_name = config.model_name
    dataset = config.data.dataset
    
    # Add regularization suffix
    reg_parts = []
    if config.weight_decay > 0:
        reg_parts.append(f"wd{config.weight_decay}")
    if config.data.random_crop:
        reg_parts.append("crop")
    if config.data.augment_flip_rotate:
        reg_parts.append("aug")

    reg_suffix = "_" + "_".join(reg_parts) if reg_parts else "_baseline"

    return f"{model_name}_{dataset}{reg_suffix}"


def print_run_card(config: ExperimentConfig, model, train_loader, test_loader, optimizer, scheduler, device):
    """
    Print a detailed run card with all experiment information.
    """
    params_info = get_model_params_info(model)

    # Get scheduler info
    scheduler_info = "None"
    if scheduler is not None:
        scheduler_type = type(scheduler).__name__
        if hasattr(scheduler, 'milestones'):
            scheduler_info = f"{scheduler_type} (milestones={list(scheduler.milestones)}, gamma={scheduler.gamma})"
        elif hasattr(scheduler, 'T_max'):
            scheduler_info = f"{scheduler_type} (T_max={scheduler.T_max})"
        elif hasattr(scheduler, 'gamma'):
            scheduler_info = f"{scheduler_type} (gamma={scheduler.gamma})"
        else:
            scheduler_info = scheduler_type

    # Calculate total training steps
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.num_epochs

    # Estimate samples seen (accounting for last batch potentially being smaller)
    approx_samples_per_epoch = len(train_loader.dataset)
    total_samples = approx_samples_per_epoch * config.num_epochs

    print("\n" + "="*80)
    print("RUN CARD - EXPERIMENT CONFIGURATION")
    print("="*80)

    print("\n┌─ EXPERIMENT METADATA ─────────────────────────────────────────────────────┐")
    print(f"│ Experiment ID        : {get_experiment_id(config):<54} │")
    print(f"│ Timestamp            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<54} │")
    print(f"│ Device               : {str(device):<54} │")
    print(f"│ Random Seed          : {config.seed:<54} │")
    print("└───────────────────────────────────────────────────────────────────────────┘")

    print("\n┌─ MODEL ARCHITECTURE ──────────────────────────────────────────────────────┐")
    print(f"│ Model Name           : {config.model_name:<54} │")
    print(f"│ Input Shape          : {str(config.input_shape):<54} │")
    print(f"│ Effective Shape      : ({config.input_shape[0]}, {config.data.crop_size}, {config.data.crop_size}){'':<39} │")
    print(f"│ Num Classes          : {config.num_classes:<54} │")
    print(f"│ Paper Params (no BN) : {params_info['paper_params']:,}{'':<54}"[:-54] + f"{params_info['paper_params']:>20,} │")
    print(f"│ Total Parameters     : {params_info['total_params']:,}{'':<54}"[:-54] + f"{params_info['total_params']:>20,} │")
    print(f"│ Trainable Params     : {params_info['trainable_params']:,}{'':<54}"[:-54] + f"{params_info['trainable_params']:>20,} │")
    print(f"│ Non-trainable Params : {params_info['non_trainable_params']:,}{'':<54}"[:-54] + f"{params_info['non_trainable_params']:>20,} │")
    print("└───────────────────────────────────────────────────────────────────────────┘")

    print("\n┌─ DATASET & DATA LOADING ──────────────────────────────────────────────────┐")
    print(f"│ Dataset              : {config.data.dataset.upper():<54} │")
    print(f"│ Train Samples        : {len(train_loader.dataset):,}{'':<54}"[:-54] + f"{len(train_loader.dataset):>20,} │")
    print(f"│ Test Samples         : {len(test_loader.dataset):,}{'':<54}"[:-54] + f"{len(test_loader.dataset):>20,} │")
    print(f"│ Batch Size           : {config.data.batch_size:<54} │")
    print(f"│ Steps per Epoch      : {steps_per_epoch:,}{'':<54}"[:-54] + f"{steps_per_epoch:>20,} │")
    print(f"│ Crop Size (always)   : 28×28 (from 32×32){'':<39} │")
    crop_type = "Random Crop" if config.data.random_crop else "Center Crop"
    print(f"│ Crop Type            : {crop_type:<54} │")
    print("└───────────────────────────────────────────────────────────────────────────┘")

    print("\n┌─ REGULARIZATION TECHNIQUES ───────────────────────────────────────────────┐")
    print(f"│ Explicit Reg OFF     : {str(config.explicit_reg_off):<54} │")
    print(f"│ Weight Decay         : {config.weight_decay:<54} │")
    print(f"│ Random Crop          : {str(config.data.random_crop):<54} │")
    print(f"│ Augment (Flip+Rotate): {str(config.data.augment_flip_rotate):<54} │")
    print("└───────────────────────────────────────────────────────────────────────────┘")

    print("\n┌─ OPTIMIZATION & TRAINING ─────────────────────────────────────────────────┐")
    print(f"│ Optimizer            : {config.optimizer.upper():<54} │")
    print(f"│ Learning Rate        : {config.learning_rate:<54} │")
    print(f"│ Momentum             : {config.momentum:<54} │")
    print(f"│ LR Scheduler         : {scheduler_info:<54} │")
    print(f"│ Num Epochs           : {config.num_epochs:<54} │")
    print(f"│ Total Training Steps : {total_steps:,}{'':<54}"[:-54] + f"{total_steps:>20,} │")
    print(f"│ Est. Samples Seen    : {total_samples:,}{'':<54}"[:-54] + f"{total_samples:>20,} │")
    early_stopping_status = f"Enabled (window={config.early_stopping_window}, min_epochs={config.early_stopping_min_epochs})" if config.early_stopping_enabled else "Disabled"
    print(f"│ Early Stopping       : {early_stopping_status:<54} │")
    print("└───────────────────────────────────────────────────────────────────────────┘")

    if config.use_wandb:
        print("\n┌─ LOGGING & TRACKING ──────────────────────────────────────────────────────┐")
        print(f"│ Weights & Biases     : Enabled{'':<49} │")
        print(f"│ W&B Project          : {config.wandb_project:<54} │")
        print(f"│ W&B Entity           : {config.wandb_entity if config.wandb_entity else 'N/A':<54} │")
        print(f"│ Save Directory       : {config.save_dir:<54} │")
        print(f"│ Log Interval         : Every {config.log_interval} batches{'':<41} │")
        print("└───────────────────────────────────────────────────────────────────────────┘")
    else:
        print("\n┌─ LOGGING & TRACKING ──────────────────────────────────────────────────────┐")
        print(f"│ Weights & Biases     : Disabled{'':<48} │")
        print(f"│ Save Directory       : {config.save_dir:<54} │")
        print(f"│ Log Interval         : Every {config.log_interval} batches{'':<41} │")
        print("└───────────────────────────────────────────────────────────────────────────┘")

    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
