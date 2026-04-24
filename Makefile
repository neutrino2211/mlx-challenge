# MLX Challenge - Training Commands
# Usage: make <target>

DATA ?= train.jsonl
MODEL ?= 50m
TOKENIZER_DIR ?= tokenizer
VOCAB_SIZE ?= 32000
RUN ?= QRK-50M
PROMPT ?= "Hello, world!"
MLX_PRETRAIN ?= mlx-pretrain

.PHONY: help setup tokenizer train-50m train-150m train generate clean

help:
	@echo "MLX Challenge Training"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Clone mlx-pretrain + install deps"
	@echo "  make tokenizer      Train tokenizer (DATA=train.jsonl)"
	@echo "  make fix-jsonl      Fix multiline JSON to proper JSONL"
	@echo ""
	@echo "Training:"
	@echo "  make train          Train model (MODEL=50m|150m)"
	@echo "  make train-50m      Train 50M model"
	@echo "  make train-150m     Train 150M model"
	@echo ""
	@echo "Inference:"
	@echo "  make generate       Generate text (RUN=name PROMPT=text)"
	@echo ""
	@echo "Options:"
	@echo "  DATA=file.jsonl     Training data file"
	@echo "  MODEL=50m|150m      Model size"
	@echo "  VOCAB_SIZE=32000    Tokenizer vocab size"
	@echo "  RUN=QRK-50M         Run name for generation"
	@echo "  PROMPT=\"text\"       Prompt for generation"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make fix-jsonl DATA=train.jsonl"
	@echo "  make tokenizer DATA=train.jsonl"
	@echo "  make train MODEL=50m"
	@echo "  make generate RUN=\"QRK-50M\" PROMPT=\"Once upon a time\""

setup: clone-mlx-pretrain install

clone-mlx-pretrain:
	@if [ ! -d "$(MLX_PRETRAIN)" ]; then \
		echo "Cloning mlx-pretrain..."; \
		git clone https://github.com/N8python/mlx-pretrain.git $(MLX_PRETRAIN); \
	else \
		echo "mlx-pretrain already exists"; \
	fi

install:
	pip install -r $(MLX_PRETRAIN)/requirements.txt
	pip install tokenizers

tokenizer:
	@test -f $(DATA) || (echo "Error: $(DATA) not found" && exit 1)
	python train_tokenizer.py --data $(DATA) --output $(TOKENIZER_DIR) --vocab-size $(VOCAB_SIZE)

train: train-$(MODEL)

train-50m: check-mlx-pretrain check-tokenizer
	python $(MLX_PRETRAIN)/train.py --config model-config-50m.yaml

train-150m: check-mlx-pretrain check-tokenizer
	python $(MLX_PRETRAIN)/train.py --config model-config-150m.yaml

generate: check-mlx-pretrain
	python $(MLX_PRETRAIN)/generate.py --run $(RUN) --prompt $(PROMPT)

check-mlx-pretrain:
	@test -d "$(MLX_PRETRAIN)" || (echo "Error: Run 'make setup' first" && exit 1)

check-tokenizer:
	@test -f $(TOKENIZER_DIR)/tokenizer.json || (echo "Error: Run 'make tokenizer' first" && exit 1)

clean:
	rm -rf runs/
	rm -rf $(TOKENIZER_DIR)/

clean-all: clean
	rm -rf $(MLX_PRETRAIN)/

# Fix multiline JSON to proper JSONL
fix-jsonl:
	@test -f $(DATA) || (echo "Error: $(DATA) not found" && exit 1)
	@echo "Fixing multiline JSON in $(DATA)..."
	@python3 -c "\
import json; \
content = open('$(DATA)', 'r').read(); \
objects = []; buffer = ''; brace_count = 0; \
[exec('brace_count += 1') if c == '{' else exec('brace_count -= 1; objects.append(json.loads(buffer)) if brace_count == 0 and buffer.strip() else None; buffer = \"\" if brace_count == 0 else buffer') if c == '}' else None for c in content for buffer in [buffer + c]]; \
exit(1) if not objects else None; \
open('$(DATA).fixed', 'w').writelines(json.dumps(o) + '\n' for o in objects); \
print(f'Converted {len(objects)} objects to proper JSONL')" 2>/dev/null || \
	python3 -c "\
import json
content = open('$(DATA)', 'r').read()
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
            except:
                pass
            buffer = ''
with open('$(DATA).fixed', 'w') as f:
    for obj in objects:
        f.write(json.dumps(obj) + '\n')
print(f'Converted {len(objects)} objects to proper JSONL')"
	@mv $(DATA) $(DATA).original
	@mv $(DATA).fixed $(DATA)
	@echo "Original saved as $(DATA).original"

# Quick test with small subset
test-tokenizer:
	@echo "Testing tokenizer..."
	python -c "from tokenizers import Tokenizer; t = Tokenizer.from_file('$(TOKENIZER_DIR)/tokenizer.json'); print('Vocab:', t.get_vocab_size()); r = t.encode('Hello <knowledge>test</knowledge>'); print('Tokens:', r.tokens)"
