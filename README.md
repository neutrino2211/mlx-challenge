# MLX Challenge Config

Configuration files for training small LLMs using [mlx-pretrain](https://github.com/N8python/mlx-pretrain) on Apple Silicon.

## Files

- `model-config-50m.yaml` - 50M parameter model (safe for 16GB, ~2-3 hours for 100M tokens)
- `model-config-150m.yaml` - 150M parameter model (~8-12 hours for 100M tokens)
- `tokenizer-config.yaml` - BPE tokenizer with thought injection special tokens

## Quick Start

```bash
make install                  # Install deps
make tokenizer DATA=train.jsonl   # Train tokenizer
make train MODEL=50m          # Train 50M model
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

# Clean up
make clean
```

### Manual (without Make)

```bash
pip install mlx-pretrain tokenizers

# Tokenizer
python train_tokenizer.py --data train.jsonl --output tokenizer/

# Train
mlx-pretrain train --config model-config-50m.yaml
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
