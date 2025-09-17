# NPO: Neural Preference Optimization for Machine Unlearning

This repository implements various machine unlearning methods including NPO (Neural Preference Optimization), influence-based unlearning, and traditional gradient-based methods.

## Project Structure

```
npo/
├── forget.py                    # Main unlearning script (R-TOFU style)
├── compute_influence.py         # Token-level influence score computation
├── trainer/                     # Custom trainer and loss functions
│   ├── __init__.py
│   ├── trainer.py              # CustomTrainerForgetting
│   └── losses.py               # All loss functions (GA, NPO, Influence, etc.)
├── token_weight/               # Token-level analysis tools
│   ├── token_level_npo.py     # Token-level NPO implementation
│   ├── select_forget_tokens.py # Token selection utilities
│   ├── demo_forget_usage.py    # Demo scripts
│   └── evaluate_unlearning.py  # Evaluation utilities
├── config/                     # Configuration files
│   ├── npo.yaml               # Main NPO configuration
│   ├── model_config.yaml      # Model configurations
│   └── ds_config/             # DeepSpeed configurations
├── scripts/                    # Execution scripts
│   ├── run_npo.sh             # Run NPO unlearning
│   ├── run_ga.sh              # Run GA unlearning
│   ├── run_influence.sh       # Run influence-based unlearning
│   └── token_weight/          # Token-level scripts
└── utils.py                   # Utility functions
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install flash-attention (optional)
pip install flash-attn --no-build-isolation
```

## Usage

### 1. NPO Unlearning

```bash
# Run NPO unlearning
bash scripts/run_npo.sh

# Or with custom parameters
python forget.py \
    model_family=llama3-8b \
    forget_loss=NPO+GD \
    lr=1e-5 \
    num_epochs=5
```

### 2. Gradient Ascent (GA) Unlearning

```bash
# Run GA unlearning
bash scripts/run_ga.sh

# Or with custom parameters
python forget.py \
    model_family=llama3-8b \
    forget_loss=GA+GD \
    lr=1e-5 \
    num_epochs=5
```

### 3. Influence-based Unlearning

```bash
# Run influence-based unlearning
bash scripts/run_influence.sh

# Or with custom parameters
python forget.py \
    model_family=llama3-8b \
    forget_loss=INFLUENCE+GD \
    lr=1e-5 \
    num_epochs=5
```

### 4. Token-level Analysis

```bash
# Compute token-level influence scores
bash scripts/token_weight/run_token_influence.sh

# Run token-level NPO
bash scripts/token_weight/run_token_npo.sh
```

## Loss Functions

The framework supports multiple loss functions that can be combined:

### Forget Losses
- **GA1/GA2/GA3**: Gradient Ascent variants
- **NPO**: Neural Preference Optimization
- **INFLUENCE**: Influence-based weighting
- **IDK1/IDK2/IDK3**: "I don't know" responses
- **FINETUNE**: Standard fine-tuning

### Regularization Losses
- **GD**: Gradient Descent on retain data
- **KL**: KL divergence with reference model

### Examples
- `GA+GD`: Gradient Ascent + Gradient Descent
- `NPO+KL`: NPO + KL divergence
- `INFLUENCE+GD`: Influence-based + Gradient Descent

## Configuration

Edit `config/npo.yaml` to customize:
- Model family and path
- Loss function combination
- Training parameters
- Data paths and splits

## Features

- **Multiple Unlearning Methods**: GA, NPO, Influence-based
- **Token-level Analysis**: Per-token influence scores
- **Flexible Loss Combinations**: Mix and match different strategies
- **R-TOFU Compatibility**: Compatible with R-TOFU dataset and evaluation
- **DeepSpeed Support**: Multi-GPU training support
- **LoRA Support**: Parameter-efficient fine-tuning

## Citation

If you use this code, please cite:

```bibtex
@article{npo2024,
  title={Neural Preference Optimization for Machine Unlearning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

# tnpo
