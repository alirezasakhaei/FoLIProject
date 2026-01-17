"""
Text classification training script using Hugging Face Transformers.
Based on: https://huggingface.co/docs/transformers/en/tasks/sequence_classification

Usage:
    python -m src.train.text_main <config.yaml>
    python -m src.train.text_main recipes/text/distilbert_imdb_true_labels.yaml
"""
import sys
import os
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback
)
import evaluate
import wandb


@dataclass
class TextExperimentConfig:
    """Configuration for text classification experiments."""

    # Experiment identification
    experiment_id: str = "text_experiment"

    # Model configuration
    model_name: str = "distilbert-base-uncased"
    num_classes: int = 2
    task_type: str = "text_classification"

    # Data configuration
    dataset: str = "imdb"
    data_root: str = "./data"
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    randomization: str = "none"  # none, random_labels
    max_length: int = 512
    data_fraction: float = 1.0

    # Training hyperparameters
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    optimizer: str = "adamw"
    weight_decay: float = 0.01

    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_min_epochs: int = 1
    early_stopping_window: int = 3

    # Logging
    use_wandb: bool = True
    wandb_project: str = "FOLI-Project"
    wandb_entity: Optional[str] = None
    save_dir: str = "./checkpoints/text"
    log_interval: int = 100
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"

    # Device
    device: str = "cuda"

    # Reproducibility
    seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TextExperimentConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TextExperimentConfig':
        """Create configuration from dictionary."""
        config = config_dict.copy()

        # Flatten nested sections
        sections_to_flatten = ['model', 'data', 'training', 'early_stopping', 'logging']
        for section in sections_to_flatten:
            if section in config and isinstance(config[section], dict):
                section_dict = config.pop(section)
                for key, value in section_dict.items():
                    if key not in config:
                        config[key] = value

        # Filter to only valid fields and convert types
        valid_fields = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        filtered_dict = {}
        for key, value in config.items():
            if key in valid_fields:
                # Convert to proper type
                expected_type = valid_fields[key]
                # Handle Optional types
                if hasattr(expected_type, '__origin__'):
                    # For Optional[X], get the actual type X
                    args = expected_type.__args__
                    expected_type = args[0] if args else expected_type

                # Convert value to expected type
                if expected_type == float and not isinstance(value, float):
                    filtered_dict[key] = float(value)
                elif expected_type == int and not isinstance(value, int):
                    filtered_dict[key] = int(value)
                elif expected_type == bool and not isinstance(value, bool):
                    filtered_dict[key] = bool(value)
                elif expected_type == str and not isinstance(value, str):
                    filtered_dict[key] = str(value)
                else:
                    filtered_dict[key] = value

        return cls(**filtered_dict)

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_dataset(config: TextExperimentConfig):
    """Load and prepare the dataset."""
    # Load dataset
    print(f"\nLoading {config.dataset} dataset...")
    dataset = load_dataset(config.dataset)

    # Apply label randomization if needed
    if config.randomization == "random_labels":
        print(f"Randomizing labels with seed {config.seed}...")
        generator = torch.Generator().manual_seed(config.seed)
        num_samples = len(dataset["train"])
        random_labels = torch.randint(
            0, config.num_classes,
            (num_samples,),
            generator=generator
        ).tolist()

        # Update labels
        dataset["train"] = dataset["train"].map(
            lambda example, idx: {"label": random_labels[idx]},
            with_indices=True
        )

    # Apply data fraction if needed
    if config.data_fraction < 1.0:
        print(f"Using {config.data_fraction*100:.1f}% of training data...")
        train_size = len(dataset["train"])
        subset_size = int(train_size * config.data_fraction)
        dataset["train"] = dataset["train"].select(range(subset_size))

    return dataset


def preprocess_function(examples, tokenizer, max_length):
    """Tokenize the text."""
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def compute_metrics(eval_pred, accuracy_metric):
    """Compute accuracy metric."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


class TrainEvalCallback(TrainerCallback):
    """Custom callback to evaluate on training set periodically."""

    def __init__(self, trainer, train_dataset, eval_freq):
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.eval_freq = eval_freq

    def on_step_end(self, args, state, control, **kwargs):
        """Evaluate on training set at the same frequency as test evaluation."""
        if state.global_step % self.eval_freq == 0 and state.global_step > 0:
            # Evaluate on training set
            train_metrics = self.trainer.evaluate(self.train_dataset, metric_key_prefix="train")

            # Log to wandb if enabled
            if args.report_to and "wandb" in args.report_to:
                wandb.log({
                    "train_accuracy": train_metrics["train_accuracy"],
                    "train_loss": train_metrics["train_loss"],
                    "step": state.global_step
                })

        return control


def main(config: TextExperimentConfig):
    """Main training loop for text classification."""

    # Validate wandb settings
    if config.use_wandb:
        if not config.wandb_project:
            raise ValueError("wandb_project must be set in config")
        if not config.wandb_entity:
            raise ValueError("wandb_entity must be set in config")

    # Set seed
    set_seed(config.seed)

    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.experiment_id,
            config=config.to_dict(),
        )

    # Load dataset
    dataset = load_and_prepare_dataset(config)

    # Load tokenizer
    print(f"\nLoading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, config.max_length),
        batched=True
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load accuracy metric
    accuracy = evaluate.load("accuracy")

    # Load model
    print(f"\nLoading model: {config.model_name}")
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_classes,
        id2label=id2label,
        label2id=label2id
    )

    # Calculate steps per epoch for evaluation
    # We want to evaluate 10 times per epoch
    num_train_samples = len(tokenized_dataset["train"])
    steps_per_epoch = num_train_samples // config.batch_size
    eval_steps = max(1, steps_per_epoch // 10)  # Evaluate 10 times per epoch

    print(f"\nTraining configuration:")
    print(f"  Total training samples: {num_train_samples}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Evaluation frequency: {eval_steps} steps (10 times per epoch)")

    # Training arguments
    # When load_best_model_at_end is True, save_strategy must match eval_strategy
    save_strategy = "steps" if config.early_stopping_enabled else config.save_strategy
    save_steps = eval_steps if config.early_stopping_enabled else None

    training_args = TrainingArguments(
        output_dir=config.save_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        eval_strategy="steps",  # Evaluate at specific step intervals
        eval_steps=eval_steps,  # Evaluate 10 times per epoch
        save_strategy=save_strategy,
        save_steps=save_steps,
        load_best_model_at_end=config.early_stopping_enabled,
        push_to_hub=False,
        logging_steps=config.log_interval,
        warmup_steps=config.warmup_steps,
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, accuracy),
    )

    # Add callback to evaluate on training set
    train_eval_callback = TrainEvalCallback(
        trainer=trainer,
        train_dataset=tokenized_dataset["train"],
        eval_freq=eval_steps
    )
    trainer.add_callback(train_eval_callback)

    # Print experiment info
    print("\n" + "="*60)
    print(f"Starting Text Classification Experiment")
    print("="*60)
    print(f"Experiment ID: {config.experiment_id}")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset}")
    print(f"Label randomization: {config.randomization}")
    print(f"Training samples: {len(tokenized_dataset['train'])}")
    print(f"Test samples: {len(tokenized_dataset['test'])}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print("="*60 + "\n")

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Test Accuracy: {test_results.get('eval_accuracy', 0)*100:.2f}%")
    print(f"Test Loss: {test_results.get('eval_loss', 0):.4f}")
    print("="*60 + "\n")

    # Save results
    results = {
        "experiment_id": config.experiment_id,
        "config": config.to_dict(),
        "test_accuracy": test_results.get('eval_accuracy', 0) * 100,
        "test_loss": test_results.get('eval_loss', 0),
        "metrics": test_results,
    }

    results_file = os.path.join(results_dir, f"{config.experiment_id}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Finish wandb run
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.train.text_main <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"\nLoading configuration from: {config_path}")
    config = TextExperimentConfig.from_yaml(config_path)

    main(config)
