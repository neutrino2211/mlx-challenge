# MLX Challenge Config

Configuration files for training small LLMs using [mlx-pretrain](https://github.com/N8python/mlx-pretrain) on Apple Silicon.

## Files

- `model-config-50m.yaml` - 50M parameter model (safe for 16GB, ~2-3 hours for 100M tokens)
- `model-config-150m.yaml` - 150M parameter model (~8-12 hours for 100M tokens)
- `tokenizer-config.yaml` - BPE tokenizer with thought injection special tokens

## Quick Start

```bash
make setup                        # Clone mlx-pretrain + install deps
make tokenizer DATA=train.jsonl   # Train tokenizer
make train MODEL=50m              # Train 50M model
```

## Setup

`mlx-pretrain` is a repo you clone, not a pip package:

```bash
# Clone mlx-pretrain into this directory
git clone https://github.com/N8python/mlx-pretrain.git mlx-pretrain

# Install dependencies (use Python 3.10 or 3.11, NOT 3.13)
pip install -r mlx-pretrain/requirements.txt
pip install tokenizers
```

## Usage

```bash
# See all commands
make help

# Train tokenizer from your data
make tokenizer DATA=my_data.jsonl
make tokenizer DATA=corpus.txt VOCAB_SIZE=16000

# Train models
make train-50m      # ~2-3 hours, safe for 16GB
make train-150m     # ~8-12 hours

# Test tokenizer works
make test-tokenizer

# Generate text
make generate RUN="QRK-50M" PROMPT="Hello world"

# Clean up
make clean
```

### Manual (without Make)

```bash
# Tokenizer (use our script)
python train_tokenizer.py --data train.jsonl --output tokenizer/

# Train (use mlx-pretrain's train.py)
python mlx-pretrain/train.py --config model-config-50m.yaml

# Generate
python mlx-pretrain/generate.py --run "QRK-50M" --prompt "Hello"
```

## Special Tokens

These configs include special tokens for **Thought Injection** - a technique where the model can request external knowledge mid-generation:

- `<knowledge>` / `</knowledge>` - Model emits to request knowledge
- `<k_res>` / `<k_end>` - Markers for injected knowledge responses

## Hardware Requirements

- Apple Silicon Mac (M1/M2/M3)
- 16GB RAM minimum
- 50M model: ~4GB memory usage
- 150M model: ~10GB memory usage with gradient checkpointing
