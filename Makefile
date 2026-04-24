# MLX Challenge - Training Commands
# Usage: make <target>

DATA ?= train.jsonl
MODEL ?= 50m
TOKENIZER_DIR ?= tokenizer
VOCAB_SIZE ?= 32000

.PHONY: help tokenizer train-50m train-150m train clean

help:
	@echo "MLX Challenge Training"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies"
	@echo "  make tokenizer      Train tokenizer (DATA=train.jsonl)"
	@echo ""
	@echo "Training:"
	@echo "  make train          Train model (MODEL=50m|150m)"
	@echo "  make train-50m      Train 50M model"
	@echo "  make train-150m     Train 150M model"
	@echo ""
	@echo "Options:"
	@echo "  DATA=file.jsonl     Training data file"
	@echo "  MODEL=50m|150m      Model size"
	@echo "  VOCAB_SIZE=32000    Tokenizer vocab size"
	@echo ""
	@echo "Examples:"
	@echo "  make tokenizer DATA=my_data.jsonl"
	@echo "  make train MODEL=50m"

install:
	pip install mlx-pretrain tokenizers

tokenizer:
	@test -f $(DATA) || (echo "Error: $(DATA) not found" && exit 1)
	python train_tokenizer.py --data $(DATA) --output $(TOKENIZER_DIR) --vocab-size $(VOCAB_SIZE)

train: train-$(MODEL)

train-50m: check-tokenizer
	mlx-pretrain train --config model-config-50m.yaml

train-150m: check-tokenizer
	mlx-pretrain train --config model-config-150m.yaml

check-tokenizer:
	@test -f $(TOKENIZER_DIR)/tokenizer.json || (echo "Error: Run 'make tokenizer' first" && exit 1)

clean:
	rm -rf runs/
	rm -rf $(TOKENIZER_DIR)/

# Quick test with small subset
test-tokenizer:
	@echo "Testing tokenizer..."
	python -c "from tokenizers import Tokenizer; t = Tokenizer.from_file('$(TOKENIZER_DIR)/tokenizer.json'); print('Vocab:', t.get_vocab_size()); r = t.encode('Hello <knowledge>test</knowledge>'); print('Tokens:', r.tokens)"
