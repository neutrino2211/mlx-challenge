#!/usr/bin/env python3
"""
Fix multiline JSON to proper JSONL format.

Usage:
    python fix_jsonl.py input.jsonl
    python fix_jsonl.py input.jsonl --output fixed.jsonl
"""

import argparse
import json
from pathlib import Path


def fix_jsonl(input_path: str, output_path: str = None):
    """Convert multiline JSON objects to proper JSONL (one object per line)."""
    
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('.jsonl.fixed')
    else:
        output_path = Path(output_path)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    objects = []
    buffer = ''
    brace_count = 0
    
    for char in content:
        buffer += char
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and buffer.strip():
                try:
                    obj = json.loads(buffer)
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                buffer = ''
    
    if not objects:
        print(f"Error: No valid JSON objects found in {input_path}")
        return False
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for obj in objects:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(objects)} objects to proper JSONL")
    print(f"Output: {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix multiline JSON to proper JSONL")
    parser.add_argument("input", help="Input file with multiline JSON")
    parser.add_argument("--output", "-o", help="Output file (default: input.jsonl.fixed)")
    
    args = parser.parse_args()
    
    success = fix_jsonl(args.input, args.output)
    
    if success and args.output is None:
        # Replace original
        input_path = Path(args.input)
        fixed_path = input_path.with_suffix('.jsonl.fixed')
        backup_path = input_path.with_suffix('.jsonl.original')
        
        input_path.rename(backup_path)
        fixed_path.rename(input_path)
        print(f"Original saved as: {backup_path}")
