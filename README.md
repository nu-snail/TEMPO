# TEMPO: Temporal Enforcement via Mode-separated Policy Optimization

TEMPO uses reinforcement learning (GRPO with LoRA) to train language models that avoid citing post-cutoff knowledge during backtesting. A two-mode reward first drives temporal leakage to zero, then optimizes prediction accuracy subject to evidence-coverage constraints. At inference, TEMPO requires a single forward pass---identical in cost to the unmodified model.

## Project Structure

```
├── QualityGatedExperiment/   # Main experiment (reward v4, paper results)
│   ├── configs/              #   YAML configs per model
│   ├── scripts/              #   Train/eval shell scripts
│   └── src/                  #   Training, evaluation, reward pipeline
├── CoverageGatedExperiment/  # Intermediate experiment (reward v2-v3)
├── FullExperiment/           # Earlier full-pipeline experiment
├── TRLExperiment/            # Early TRL-based prototype
├── src/                      # Shared modules (evaluation, search, prompts)
├── scripts/                  # Top-level orchestration scripts
├── configs/                  # Top-level configs (Qwen, DeepSeek)
├── dataset/                  # Data splits (not tracked; see below)
└── tests/                    # Integration tests
```

## Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- API keys: [OpenRouter](https://openrouter.ai/) and [Tinker](https://pypi.org/project/tinker/)
- GPU with ≥24 GB VRAM for training (A100/H100 recommended)

## Setup

```bash
git clone https://github.com/<your-org>/TEMPO.git && cd TEMPO

# Install dependencies
uv sync

# Configure API keys
cp QualityGatedExperiment/.env.example .env
# Edit .env and set OPENROUTER_API_KEY and TINKER_API_KEY
```

## Data

The `dataset/` directory is not tracked due to size. It contains three splits:

| Split | Contents |
|-------|----------|
| `dataset/train/` | Stock rankings, salary predictions, legal cases (training instances) |
| `dataset/evaluation/` | Held-out evaluation instances per task |
| `dataset/control/` | Post-cutoff instances for zero-knowledge baseline |

Each experiment copies data into its own `runtime/data/` via `scripts/bootstrap_data.sh`.

## Running Experiments

### Reproduce paper results (QualityGatedExperiment)

This is the final experiment configuration (reward v4) used to produce all results reported in the paper.

```bash
cd QualityGatedExperiment

# Copy data into runtime/
bash scripts/bootstrap_data.sh

# Copy and set API keys
cp .env.example .env   # then edit

# Full pipeline: train + evaluate all methods (base, best, final, RAG, TimeSPEC)
# Qwen3-30B-A3B on all three tasks:
bash scripts/run_full.sh configs/train_qwen3_30b.yaml configs/eval_qwen3_30b.yaml stock salary legal

# Qwen3.5-35B-A3B:
bash scripts/run_full.sh configs/train_qwen35_35b.yaml configs/eval_qwen35_35b.yaml stock salary legal
```

Each task runs: train → evaluate (base + best + final checkpoints + RAG + TimeSPEC) → cleanup.

Results are written to `runtime/results/<model>/`.

### Run a single task

```bash
cd QualityGatedExperiment

# Train on stock only
bash scripts/run_full.sh configs/train_qwen3_30b.yaml configs/eval_qwen3_30b.yaml stock
```

### Run individual steps

```bash
cd QualityGatedExperiment
export PYTHONPATH=src:$PYTHONPATH && source .env

# Train
python -m quality_gated_experiment.train \
    --config configs/train_qwen3_30b.yaml \
    --train_tasks stock \
    --log_dir runtime/results/train_stock

# Evaluate base model
python -m quality_gated_experiment.evaluate \
    --config configs/eval_qwen3_30b.yaml \
    --variant base \
    --trained_task stock \
    --eval_tasks stock \
    --output_dir runtime/results/eval_stock_base

# Evaluate RAG baseline
python -m quality_gated_experiment.rag_evaluate \
    --config configs/eval_rag_qwen3_30b.yaml \
    --eval_tasks stock \
    --output_dir runtime/results/eval_stock_rag

# Evaluate TimeSPEC baseline
python -m quality_gated_experiment.timespec_evaluate \
    --config configs/eval_timespec_qwen3_30b.yaml \
    --eval_tasks stock \
    --output_dir runtime/results/eval_stock_timespec
```

### Earlier experiment versions

The `CoverageGatedExperiment/`, `FullExperiment/`, and `TRLExperiment/` directories contain earlier reward design iterations (v1--v3). They follow the same train/eval pattern:

```bash
cd CoverageGatedExperiment
bash scripts/bootstrap_data.sh
bash scripts/run_full.sh stock salary legal
```

## Configuration

Training and evaluation are controlled by YAML configs in each experiment's `configs/` directory. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | HuggingFace model ID | `Qwen/Qwen3-30B-A3B` |
| `lora_rank` | LoRA rank | 32 |
| `group_size` | GRPO group size | 16 |
| `batch_size` | Training batch size | 8 |
| `num_epochs` | Training epochs | 2 |
| `judge_model` | LLM for claim verification | `google/gemini-3-flash-preview` |
| `search_model` | Search API model | `perplexity/sonar` |

## License

TBD
