# MLX Challenge Config

Configuration files for training small LLMs using [mlx-pretrain](https://github.com/N8python/mlx-pretrain) on Apple Silicon.

## Files

- `model-config-50m.yaml` - 50M parameter model (safe for 16GB, ~2-3 hours for 100M tokens)
- `model-config-150m.yaml` - 150M parameter model (~8-12 hours for 100M tokens)
- `tokenizer-config.yaml` - BPE tokenizer with thought injection special tokens

## Usage

1. Install mlx-pretrain:
```bash
pip install mlx-pretrain
```

2. Train tokenizer:
```bash
mlx-pretrain tokenizer --config tokenizer-config.yaml
```

3. Train model (start with 50M to validate setup):
```bash
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
