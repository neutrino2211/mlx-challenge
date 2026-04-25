#!/usr/bin/env python3
"""
QRK Training Pipeline on Modal

Full pipeline:
1. Prepare pretrain data (70% FineWeb-Edu + 30% Hermes/Capybara)
2. Train tokenizer
3. Pretrain 300M model
4. Convert to mlx-lm format
5. Finetune on ChatML (Hermes+Capybara)
6. Finetune on TI (v2b dataset)
7. Upload to HuggingFace

Usage:
    modal run modal_train.py
    modal run modal_train.py --upload --hf-repo qrk-labs/QRK-300M-TI
"""
import modal

app = modal.App("qrk-training")

# Base image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install([
        "mlx",
        "mlx-lm[train]",
        "tokenizers",
        "datasets",
        "safetensors",
        "numpy",
        "pyyaml",
        "huggingface_hub",
    ])
    .run_commands([
        "git clone https://github.com/N8python/mlx-pretrain.git /mlx-pretrain",
    ])
)

# Persistent volume for checkpoints and data
volume = modal.Volume.from_name("qrk-training-vol", create_if_missing=True)

# HuggingFace secrets (set via `modal secret create`)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])


# =============================================================================
# Step 1: Prepare Pretrain Data (70% FineWeb-Edu + 30% Chat)
# =============================================================================
@app.function(
    image=image,
    timeout=3600 * 2,
    volumes={"/data": volume},
    memory=32768,  # 32GB for large dataset processing
)
def prepare_pretrain_data(
    total_samples: int = 500_000,
    fineweb_ratio: float = 0.7,
):
    """Prepare mixed pretraining data: FineWeb-Edu + Hermes/Capybara as raw text."""
    import json
    import random
    from pathlib import Path
    from datasets import load_dataset
    
    random.seed(42)
    output_path = Path("/data/pretrain_data.jsonl")
    
    samples = []
    
    # --- FineWeb-Edu (70%) ---
    fineweb_count = int(total_samples * fineweb_ratio)
    print(f"Loading {fineweb_count} samples from FineWeb-Edu...")
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split=f"train[:{fineweb_count}]",
            trust_remote_code=True,
        )
        for item in ds:
            if item.get("text"):
                samples.append({"text": item["text"], "source": "fineweb"})
        print(f"  Loaded {len(samples)} from FineWeb-Edu")
    except Exception as e:
        print(f"  FineWeb-Edu failed: {e}")
    
    # --- Hermes + Capybara as raw text (30%) ---
    chat_count = total_samples - fineweb_count
    hermes_count = int(chat_count * 0.7)
    capybara_count = chat_count - hermes_count
    
    def chat_to_raw_text(conversations):
        """Convert chat format to raw text (no special tokens for pretraining)."""
        parts = []
        for turn in conversations:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ["human", "user"]:
                parts.append(f"User: {content}")
            elif role in ["gpt", "assistant", "bot"]:
                parts.append(f"Assistant: {content}")
            elif role == "system":
                parts.append(f"System: {content}")
        return "\n\n".join(parts)
    
    print(f"Loading {hermes_count} samples from Hermes...")
    try:
        ds = load_dataset(
            "teknium/OpenHermes-2.5",
            split=f"train[:{hermes_count}]",
            trust_remote_code=True,
        )
        for item in ds:
            convs = item.get("conversations", [])
            if convs:
                text = chat_to_raw_text(convs)
                if text:
                    samples.append({"text": text, "source": "hermes"})
        print(f"  Total samples now: {len(samples)}")
    except Exception as e:
        print(f"  Hermes failed: {e}")
    
    print(f"Loading {capybara_count} samples from Capybara...")
    try:
        ds = load_dataset(
            "LDJnr/Capybara",
            split=f"train[:{capybara_count}]",
            trust_remote_code=True,
        )
        for item in ds:
            convs = item.get("conversation", [])
            if convs:
                text = chat_to_raw_text(convs)
                if text:
                    samples.append({"text": text, "source": "capybara"})
        print(f"  Total samples now: {len(samples)}")
    except Exception as e:
        print(f"  Capybara failed: {e}")
    
    # Shuffle
    random.shuffle(samples)
    
    # Write
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    # Stats
    fineweb_n = sum(1 for s in samples if s["source"] == "fineweb")
    hermes_n = sum(1 for s in samples if s["source"] == "hermes")
    capybara_n = sum(1 for s in samples if s["source"] == "capybara")
    
    print(f"\nPretrain data ready:")
    print(f"  FineWeb-Edu: {fineweb_n}")
    print(f"  Hermes: {hermes_n}")
    print(f"  Capybara: {capybara_n}")
    print(f"  Total: {len(samples)}")
    print(f"  Output: {output_path}")
    
    volume.commit()
    return f"Prepared {len(samples)} pretrain samples"


# =============================================================================
# Step 2: Train Tokenizer
# =============================================================================
@app.function(
    image=image,
    timeout=3600,
    volumes={"/data": volume},
)
def train_tokenizer(vocab_size: int = 32000):
    """Train BPE tokenizer with ChatML + TI special tokens."""
    import json
    from pathlib import Path
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
    
    data_path = Path("/data/pretrain_data.jsonl")
    output_dir = Path("/data/tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    special_tokens = [
        "<unk>", "<s>", "</s>", "<pad>",
        "<|im_start|>", "<|im_end|>",
        "<knowledge>", "</knowledge>", "<k_res>", "<k_end>",
    ]
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Extract text
    texts = []
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line)
            if "text" in obj:
                texts.append(obj["text"])
    
    print(f"Training tokenizer on {len(texts)} samples...")
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    
    tokenizer.save(str(output_dir / "tokenizer.json"))
    
    # HF-compatible config with chat template
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        "model_max_length": 2048,
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Tokenizer saved with vocab_size={tokenizer.get_vocab_size()}")
    
    volume.commit()
    return f"Tokenizer ready: vocab_size={tokenizer.get_vocab_size()}"


# =============================================================================
# Step 3: Pretrain
# =============================================================================
@app.function(
    image=image,
    gpu="a100",
    timeout=3600 * 24,  # 24 hours max
    volumes={"/data": volume},
)
def pretrain(
    model_name: str = "QRK-300M",
    hidden_size: int = 1536,
    num_layers: int = 8,
    num_heads: int = 12,
    num_kv_heads: int = 4,
    intermediate_size: int = 4096,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
):
    """Pretrain the base model."""
    import subprocess
    import yaml
    from pathlib import Path
    
    config = {
        "name": model_name,
        "overwrite": True,
        "data": {
            "input_file": "/data/pretrain_data.jsonl",
            "tokenizer_path": "/data/tokenizer",
            "preprocessing": {
                "max_context_size": 2048,
                "chunk_overlap": 0,
            },
            "tokenizer": {
                "normal_vocab_size": 256,
                "special_tokens": {
                    "pad": "<pad>",
                    "bos": "<s>",
                    "eos": "</s>",
                    "im_start": "<|im_start|>",
                    "im_end": "<|im_end|>",
                    "knowledge_start": "<knowledge>",
                    "knowledge_end": "</knowledge>",
                    "k_res": "<k_res>",
                    "k_end": "<k_end>",
                },
            },
        },
        "model": {
            "architecture": "llama",
            "dimensions": {
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_layers": num_layers,
            },
            "attention": {
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": None,
                "max_position_embeddings": 2048,
            },
            "normalization": {"rms_norm_eps": 1e-6},
            "rope": {"theta": 10000, "traditional": False, "scaling": None},
            "misc": {
                "attention_bias": False,
                "mlp_bias": False,
                "tie_word_embeddings": True,
            },
        },
        "training": {
            "epochs": epochs,
            "hyperparameters": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": 0.1,
            },
            "scheduler": {
                "type": "cosine_with_warmup",
                "min_lr_ratio": 0.1,
                "warmup_steps": 1000,
            },
            "optimization": {"optimizer": "adamw"},
        },
        "logging": {
            "log_dir": "/data/logs",
            "checkpoint_dir": "/data/checkpoints",
            "steps": {
                "logging_interval": 100,
                "checkpoint_interval": 5000,
                "validation_interval": 2000,
            },
            "metrics": {
                "log_loss": True,
                "log_perplexity": True,
                "log_tokens_per_second": True,
                "log_learning_rate": True,
                "log_tokens_processed": True,
            },
        },
        "system": {"seed": 42, "device": "gpu"},
    }
    
    config_path = Path("/data/pretrain_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"Starting pretraining: {model_name}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads} (kv: {num_kv_heads})")
    
    subprocess.run([
        "python", "/mlx-pretrain/train.py",
        "--config", str(config_path)
    ], check=True, cwd="/data")
    
    volume.commit()
    return f"Pretrained {model_name}"


# =============================================================================
# Step 4: Convert to mlx-lm format
# =============================================================================
@app.function(
    image=image,
    gpu="a100",
    timeout=3600,
    volumes={"/data": volume},
)
def convert_to_mlx_lm(model_name: str = "QRK-300M", num_kv_heads: int = 4):
    """Convert pretrained model to mlx-lm format with GQA fix."""
    import subprocess
    import json
    from pathlib import Path
    
    out_path = Path(f"/data/{model_name}-mlx")
    
    subprocess.run([
        "python", "/mlx-pretrain/convert-to-mlx-lm.py",
        "--run", model_name,
        "--out-path", str(out_path)
    ], check=True, cwd="/data")
    
    # Fix GQA config
    config_path = out_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    if config.get("num_key_value_heads") != num_kv_heads:
        config["num_key_value_heads"] = num_kv_heads
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Fixed num_key_value_heads={num_kv_heads}")
    
    volume.commit()
    return f"Converted to {out_path}"


# =============================================================================
# Step 5: Prepare ChatML finetuning data
# =============================================================================
@app.function(
    image=image,
    timeout=3600,
    volumes={"/data": volume},
)
def prepare_chatml_data(num_samples: int = 100_000, hermes_ratio: float = 0.7):
    """Prepare ChatML-formatted finetuning data from Hermes + Capybara."""
    import json
    import random
    from pathlib import Path
    from datasets import load_dataset
    
    random.seed(42)
    output_dir = Path("/data/finetune_chatml")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_chatml(conversations):
        output = ""
        for turn in conversations:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ["human", "user"]:
                role = "user"
            elif role in ["gpt", "assistant", "bot"]:
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                continue
            output += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return output.strip()
    
    samples = []
    
    # Hermes
    hermes_count = int(num_samples * hermes_ratio)
    print(f"Loading {hermes_count} from Hermes...")
    try:
        ds = load_dataset("teknium/OpenHermes-2.5", split=f"train[:{hermes_count}]")
        for item in ds:
            convs = item.get("conversations", [])
            if convs:
                text = format_chatml(convs)
                if text:
                    samples.append({"text": text})
    except Exception as e:
        print(f"Hermes failed: {e}")
    
    # Capybara
    capybara_count = num_samples - hermes_count
    print(f"Loading {capybara_count} from Capybara...")
    try:
        ds = load_dataset("LDJnr/Capybara", split=f"train[:{capybara_count}]")
        for item in ds:
            convs = item.get("conversation", [])
            if convs:
                text = format_chatml(convs)
                if text:
                    samples.append({"text": text})
    except Exception as e:
        print(f"Capybara failed: {e}")
    
    random.shuffle(samples)
    
    # Split
    val_count = int(len(samples) * 0.05)
    train_samples = samples[val_count:]
    val_samples = samples[:val_count]
    
    with open(output_dir / "train.jsonl", "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")
    
    with open(output_dir / "valid.jsonl", "w") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"ChatML data: {len(train_samples)} train, {len(val_samples)} val")
    
    volume.commit()
    return f"Prepared {len(train_samples)} ChatML samples"


# =============================================================================
# Step 6: Finetune on ChatML
# =============================================================================
@app.function(
    image=image,
    gpu="a100",
    timeout=3600 * 4,
    volumes={"/data": volume},
)
def finetune_chatml(
    model_name: str = "QRK-300M",
    num_layers: int = 8,
    iters: int = 2000,
    batch_size: int = 4,
):
    """LoRA finetune on ChatML data."""
    import subprocess
    
    print(f"Finetuning {model_name} on ChatML...")
    subprocess.run([
        "python", "-m", "mlx_lm.lora",
        "--model", f"/data/{model_name}-mlx",
        "--data", "/data/finetune_chatml",
        "--train",
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--num-layers", str(num_layers),
        "--adapter-path", f"/data/{model_name}-chatml-adapters",
    ], check=True)
    
    # Fuse
    print("Fusing ChatML adapters...")
    subprocess.run([
        "python", "-m", "mlx_lm.fuse",
        "--model", f"/data/{model_name}-mlx",
        "--adapter-path", f"/data/{model_name}-chatml-adapters",
        "--save-path", f"/data/{model_name}-chatml",
    ], check=True)
    
    volume.commit()
    return f"ChatML finetuned: /data/{model_name}-chatml"


# =============================================================================
# Step 7: Prepare TI data from HuggingFace
# =============================================================================
@app.function(
    image=image,
    timeout=3600,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def prepare_ti_data(dataset_name: str = "qrk-labs/akeel-ti-3k-v2b-proper"):
    """Download and prepare TI dataset from HuggingFace."""
    import json
    from pathlib import Path
    from datasets import load_dataset
    import random
    
    random.seed(42)
    output_dir = Path("/data/finetune_ti")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    
    samples = [{"text": item["text"]} for item in ds if item.get("text")]
    random.shuffle(samples)
    
    # Split
    val_count = max(1, int(len(samples) * 0.05))
    train_samples = samples[val_count:]
    val_samples = samples[:val_count]
    
    with open(output_dir / "train.jsonl", "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")
    
    with open(output_dir / "valid.jsonl", "w") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"TI data: {len(train_samples)} train, {len(val_samples)} val")
    
    volume.commit()
    return f"Prepared {len(train_samples)} TI samples"


# =============================================================================
# Step 8: Finetune on TI
# =============================================================================
@app.function(
    image=image,
    gpu="a100",
    timeout=3600 * 2,
    volumes={"/data": volume},
)
def finetune_ti(
    model_name: str = "QRK-300M",
    num_layers: int = 8,
    iters: int = 1000,
    batch_size: int = 2,
):
    """LoRA finetune on Thought Injection data."""
    import subprocess
    
    print(f"Finetuning {model_name}-chatml on TI data...")
    subprocess.run([
        "python", "-m", "mlx_lm.lora",
        "--model", f"/data/{model_name}-chatml",
        "--data", "/data/finetune_ti",
        "--train",
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--num-layers", str(num_layers),
        "--adapter-path", f"/data/{model_name}-ti-adapters",
    ], check=True)
    
    # Fuse final model
    print("Fusing TI adapters...")
    subprocess.run([
        "python", "-m", "mlx_lm.fuse",
        "--model", f"/data/{model_name}-chatml",
        "--adapter-path", f"/data/{model_name}-ti-adapters",
        "--save-path", f"/data/{model_name}-final",
    ], check=True)
    
    volume.commit()
    return f"Final model: /data/{model_name}-final"


# =============================================================================
# Step 9: Upload to HuggingFace
# =============================================================================
@app.function(
    image=image,
    timeout=3600,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def upload_to_hf(model_name: str = "QRK-300M", hf_repo: str = "qrk-labs/QRK-300M-TI"):
    """Upload final model to HuggingFace."""
    from huggingface_hub import HfApi, login
    import os
    
    login(token=os.environ["HF_TOKEN"])
    
    api = HfApi()
    api.upload_folder(
        folder_path=f"/data/{model_name}-final",
        repo_id=hf_repo,
        repo_type="model",
    )
    
    return f"Uploaded to https://huggingface.co/{hf_repo}"


# =============================================================================
# Main Pipeline
# =============================================================================
@app.local_entrypoint()
def main(
    # Data params
    pretrain_samples: int = 500_000,
    chatml_samples: int = 100_000,
    ti_dataset: str = "qrk-labs/akeel-ti-3k-v2b-proper",
    
    # Model params
    model_name: str = "QRK-300M",
    hidden_size: int = 1536,
    num_layers: int = 8,
    num_heads: int = 12,
    num_kv_heads: int = 4,
    
    # Training params
    pretrain_epochs: int = 1,
    chatml_iters: int = 2000,
    ti_iters: int = 1000,
    
    # Upload
    upload: bool = False,
    hf_repo: str = "qrk-labs/QRK-300M-TI",
    
    # Skip steps (for resuming)
    skip_pretrain_data: bool = False,
    skip_tokenizer: bool = False,
    skip_pretrain: bool = False,
    skip_convert: bool = False,
    skip_chatml: bool = False,
    skip_ti: bool = False,
):
    """
    Full QRK training pipeline.
    
    Examples:
        # Full run
        modal run modal_train.py
        
        # Resume from ChatML finetuning
        modal run modal_train.py --skip-pretrain-data --skip-tokenizer --skip-pretrain --skip-convert
        
        # Upload when done
        modal run modal_train.py --upload --hf-repo qrk-labs/QRK-300M-TI
    """
    
    if not skip_pretrain_data:
        print("=== Step 1: Prepare Pretrain Data ===")
        print(prepare_pretrain_data.remote(pretrain_samples))
    
    if not skip_tokenizer:
        print("\n=== Step 2: Train Tokenizer ===")
        print(train_tokenizer.remote())
    
    if not skip_pretrain:
        print("\n=== Step 3: Pretrain ===")
        print(pretrain.remote(
            model_name=model_name,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            epochs=pretrain_epochs,
        ))
    
    if not skip_convert:
        print("\n=== Step 4: Convert to mlx-lm ===")
        print(convert_to_mlx_lm.remote(model_name, num_kv_heads))
    
    if not skip_chatml:
        print("\n=== Step 5: Prepare ChatML Data ===")
        print(prepare_chatml_data.remote(chatml_samples))
        
        print("\n=== Step 6: Finetune ChatML ===")
        print(finetune_chatml.remote(model_name, num_layers, chatml_iters))
    
    if not skip_ti:
        print("\n=== Step 7: Prepare TI Data ===")
        print(prepare_ti_data.remote(ti_dataset))
        
        print("\n=== Step 8: Finetune TI ===")
        print(finetune_ti.remote(model_name, num_layers, ti_iters))
    
    if upload:
        print("\n=== Step 9: Upload to HuggingFace ===")
        print(upload_to_hf.remote(model_name, hf_repo))
    
    print("\n=== Pipeline Complete ===")
    print(f"Final model: /data/{model_name}-final")
