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
	@echo "  make tokenizer DATA=my_data.jsonl"
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

# Quick test with small subset
test-tokenizer:
	@echo "Testing tokenizer..."
	python -c "from tokenizers import Tokenizer; t = Tokenizer.from_file('$(TOKENIZER_DIR)/tokenizer.json'); print('Vocab:', t.get_vocab_size()); r = t.encode('Hello <knowledge>test</knowledge>'); print('Tokens:', r.tokens)"
