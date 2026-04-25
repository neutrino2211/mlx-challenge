#!/usr/bin/env python3
"""
Prepare ChatML finetuning dataset from OpenHermes and Capybara.

Usage:
    python prepare_chatml.py --samples 50000 --output chatml_train.jsonl
    python prepare_chatml.py --samples 10000 --hermes-only
    python prepare_chatml.py --samples 10000 --capybara-only
"""

import argparse
import json
import random
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: Please install datasets: pip install datasets")
    exit(1)


def format_chatml(conversations: list) -> str:
    """Convert conversation list to ChatML format string."""
    output = ""
    
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        
        # Normalize role names
        if role in ["human", "user"]:
            role = "user"
        elif role in ["gpt", "assistant", "bot"]:
            role = "assistant"
        elif role in ["system"]:
            role = "system"
        else:
            continue  # Skip unknown roles
            
        output += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    return output.strip()


def load_hermes(num_samples: int) -> list:
    """Load samples from OpenHermes 2.5."""
    print(f"Loading {num_samples} samples from OpenHermes 2.5...")
    
    try:
        ds = load_dataset(
            "teknium/OpenHermes-2.5", 
            split=f"train[:{num_samples}]",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: Could not load OpenHermes: {e}")
        return []
    
    samples = []
    for item in ds:
        convs = item.get("conversations", [])
        if convs:
            text = format_chatml(convs)
            if text:
                samples.append({"text": text, "source": "hermes"})
    
    print(f"  Loaded {len(samples)} valid samples from Hermes")
    return samples


def load_capybara(num_samples: int) -> list:
    """Load samples from Capybara."""
    print(f"Loading {num_samples} samples from Capybara...")
    
    try:
        ds = load_dataset(
            "LDJnr/Capybara",
            split=f"train[:{num_samples}]",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: Could not load Capybara: {e}")
        return []
    
    samples = []
    for item in ds:
        convs = item.get("conversation", [])
        if convs:
            text = format_chatml(convs)
            if text:
                samples.append({"text": text, "source": "capybara"})
    
    print(f"  Loaded {len(samples)} valid samples from Capybara")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare ChatML finetuning dataset")
    parser.add_argument("--samples", "-n", type=int, default=50000,
                        help="Total samples to generate (split between sources)")
    parser.add_argument("--output", "-o", default="chatml_train.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--hermes-only", action="store_true",
                        help="Only use OpenHermes")
    parser.add_argument("--capybara-only", action="store_true", 
                        help="Only use Capybara")
    parser.add_argument("--hermes-ratio", type=float, default=0.7,
                        help="Ratio of Hermes samples (default 0.7)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Validation split ratio (default 0.05)")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    all_samples = []
    
    if args.hermes_only:
        all_samples = load_hermes(args.samples)
    elif args.capybara_only:
        all_samples = load_capybara(args.samples)
    else:
        # Split between sources
        hermes_count = int(args.samples * args.hermes_ratio)
        capybara_count = args.samples - hermes_count
        
        all_samples.extend(load_hermes(hermes_count))
        all_samples.extend(load_capybara(capybara_count))
    
    if not all_samples:
        print("Error: No samples loaded!")
        return
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Split train/val
    val_count = int(len(all_samples) * args.val_split)
    train_samples = all_samples[val_count:]
    val_samples = all_samples[:val_count]
    
    # Write train
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nWrote {len(train_samples)} training samples to {output_path}")
    
    # Write val
    if val_samples:
        val_path = output_path.with_stem(output_path.stem + "_val")
        with open(val_path, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Wrote {len(val_samples)} validation samples to {val_path}")
    
    # Stats
    print(f"\nDataset stats:")
    hermes_count = sum(1 for s in all_samples if s["source"] == "hermes")
    capybara_count = sum(1 for s in all_samples if s["source"] == "capybara")
    print(f"  Hermes: {hermes_count}")
    print(f"  Capybara: {capybara_count}")
    
    # Sample preview
    print(f"\nSample preview:")
    print("-" * 50)
    print(train_samples[0]["text"][:500] + "...")


if __name__ == "__main__":
    main()
