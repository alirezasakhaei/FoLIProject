# Fundametals of Learning and Inference Final Project

A reproduction of [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530) paper

## Setup

### Prerequisites
- Python 3.10
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Create a virtual environment with Python 3.10:
```bash
uv venv --python 3.10
```

2. Activate the environment:
```bash
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

### Image Classification Experiments

```bash
./run_experiment.sh recipes/table1/alexnet_crop_no_wd_yes.yaml
```

### Text Classification Experiments

```bash
./run_text_experiment.sh recipes/text/distilbert_imdb_true_labels.yaml
```

## Configuration

Experiments are configured via YAML files in the [recipes/](recipes/) directory:
- [recipes/table1/](recipes/table1/) - Table 1 experiments
- [recipes/figure1a/](recipes/figure1a/) - Figure 1a experiments
- [recipes/figure1b/](recipes/figure1b/) - Figure 1b experiments
- [recipes/text/](recipes/text/) - Text classification experiments
- [recipes/imagenet/](recipes/imagenet/) - ImageNet experiments
- [recipes/cifar100/](recipes/cifar100/) - CIFAR-100 experiments

### WandB Setup

All experiments require WandB logging. Set your API key:

```bash
export WANDB_API_KEY=<your-wandb-api-key>
```

Configure in your YAML:

```yaml
logging:
  use_wandb: true
  wandb_project: your-project-name
  wandb_entity: your-username
```

## Project Structure

```
.
├── recipes/          # Experiment configurations
├── src/
│   ├── config.py     # Configuration management
│   ├── dataset/      # Dataset loading and transforms
│   └── train/        # Training scripts
│       ├── main.py      # Image classification training
│       └── text_main.py # Text classification training
└── data/             # Dataset storage (auto-created)
```
