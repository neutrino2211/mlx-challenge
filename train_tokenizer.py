#!/usr/bin/env python3
"""
Train a BPE tokenizer for MLX models.

Usage:
    python train_tokenizer.py --data train.jsonl --output tokenizer/
    python train_tokenizer.py --data train.txt --output tokenizer/ --vocab-size 16000
"""

import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


def extract_text_from_jsonl(path: Path) -> list[str]:
    """Extract text field from JSONL for training.
    
    Handles both proper JSONL and malformed JSONL with multiline strings.
    """
    texts = []
    
    with open(path, encoding="utf-8") as f:
        content = f.read()
    
    # First, try line-by-line (proper JSONL)
    lines = content.split('\n')
    line_by_line_works = True
    
    for line in lines[:10]:  # Test first 10 lines
        if line.strip():
            try:
                json.loads(line)
            except json.JSONDecodeError:
                line_by_line_works = False
                break
    
    if line_by_line_works:
        # Standard JSONL processing
        print("Detected: proper JSONL format")
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                for key in ["text", "content", "prompt", "completion"]:
                    if key in obj:
                        texts.append(obj[key])
                        break
            except json.JSONDecodeError:
                pass
    else:
        # Multiline JSON objects - need to parse differently
        print("Detected: multiline JSON objects, reconstructing...")
        
        # Accumulate lines until we get valid JSON
        buffer = ""
        obj_count = 0
        
        for line in lines:
            buffer += line + "\n"
            
            # Try to parse when we see what looks like object end
            if line.strip().endswith('}') or line.strip().endswith('},'):
                # Clean trailing comma if present
                test_buffer = buffer.rstrip().rstrip(',')
                try:
                    obj = json.loads(test_buffer)
                    for key in ["text", "content", "prompt", "completion"]:
                        if key in obj:
                            texts.append(obj[key])
                            obj_count += 1
                            break
                    buffer = ""
                except json.JSONDecodeError:
                    # Not complete yet, keep accumulating
                    pass
        
        print(f"Reconstructed {obj_count} JSON objects")
    
    return texts


def train_tokenizer(
    data_path: str,
    output_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
):
    """Train BPE tokenizer with thought injection special tokens."""
    
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Special tokens - order matters for IDs
    special_tokens = [
        "<unk>",          # 0
        "<s>",            # 1 - BOS
        "</s>",           # 2 - EOS  
        "<pad>",          # 3
        # Thought Injection tokens
        "<knowledge>",    # 4 - Model requests knowledge
        "</knowledge>",   # 5 - End request
        "<k_res>",        # 6 - Injected response start
        "<k_end>",        # 7 - Injected response end
        # Code tokens (optional, for code tasks)
        "<code>",         # 8
        "</code>",        # 9
        "<output>",       # 10
        "</output>",      # 11
    ]
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Handle different input formats
    if data_path.suffix == ".jsonl":
        print(f"Extracting text from JSONL: {data_path}")
        texts = extract_text_from_jsonl(data_path)
        print(f"Found {len(texts)} text samples")
        tokenizer.train_from_iterator(texts, trainer=trainer)
    else:
        # Plain text file
        print(f"Training from text file: {data_path}")
        tokenizer.train(files=[str(data_path)], trainer=trainer)
    
    # Post-processor: add BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", 1),
            ("</s>", 2),
        ],
    )
    
    # Save
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer to {tokenizer_path}")
    
    # Also save config for HuggingFace compatibility
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": 1024,
    }
    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Print some stats
    print(f"\nVocab size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {special_tokens}")
    
    # Test encoding
    test_text = "Hello world! <knowledge>What is Python?</knowledge>"
    encoded = tokenizer.encode(test_text)
    print(f"\nTest encoding:")
    print(f"  Input: {test_text}")
    print(f"  Tokens: {encoded.tokens[:20]}...")
    print(f"  IDs: {encoded.ids[:20]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for MLX")
    parser.add_argument("--data", "-d", required=True, help="Training data (JSONL or TXT)")
    parser.add_argument("--output", "-o", default="tokenizer", help="Output directory")
    parser.add_argument("--vocab-size", "-v", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--min-frequency", "-m", type=int, default=2, help="Min token frequency")
    
    args = parser.parse_args()
    
    train_tokenizer(
        data_path=args.data,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
