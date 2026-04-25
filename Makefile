# MLX Challenge - Training Commands
# Usage: make <target>

DATA ?= train.jsonl
MODEL ?= 50m
TOKENIZER_DIR ?= tokenizer
VOCAB_SIZE ?= 32000
RUN ?= QRK-50M
PROMPT ?= "Hello, world!"
MLX_PRETRAIN ?= mlx-pretrain

.PHONY: help setup tokenizer train-50m train-150m train generate clean fix-jsonl prepare-chatml finetune

help:
	@echo "MLX Challenge Training"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Clone mlx-pretrain + install deps"
	@echo "  make tokenizer      Train tokenizer (DATA=train.jsonl)"
	@echo "  make fix-jsonl      Fix multiline JSON to proper JSONL"
	@echo ""
	@echo "Pretraining:"
	@echo "  make train          Train model (MODEL=50m|150m|300m)"
	@echo "  make train-50m      Train 50M model"
	@echo "  make train-150m     Train 150M model"
	@echo "  make train-300m     Train 300M model"
	@echo ""
	@echo "Finetuning:"
	@echo "  make prepare-chatml Prepare ChatML dataset from Hermes+Capybara"
	@echo "  make convert-model  Convert pretrained model to mlx-lm format"
	@echo "  make finetune       LoRA finetune on ChatML data"
	@echo "  make fuse-lora      Fuse LoRA weights into final model"
	@echo ""
	@echo "Inference:"
	@echo "  make generate       Generate text (RUN=name PROMPT=text)"
	@echo ""
	@echo "Options:"
	@echo "  DATA=file.jsonl       Training data file"
	@echo "  MODEL=50m|150m|300m   Model size for pretraining"
	@echo "  VOCAB_SIZE=32000      Tokenizer vocab size"
	@echo "  CHATML_SAMPLES=50000  Samples for ChatML dataset"
	@echo "  FINETUNE_MODEL=QRK-300M  Model to finetune"
	@echo "  FINETUNE_ITERS=1000   Finetuning iterations"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make train MODEL=300m"
	@echo "  make prepare-chatml CHATML_SAMPLES=20000"
	@echo "  make convert-model FINETUNE_MODEL=QRK-300M"
	@echo "  make finetune"
	@echo "  make fuse-lora"

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

train-300m: check-mlx-pretrain check-tokenizer
	python $(MLX_PRETRAIN)/train.py --config model-config-300m.yaml

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

fix-jsonl:
	@test -f $(DATA) || (echo "Error: $(DATA) not found" && exit 1)
	python fix_jsonl.py $(DATA)

# Finetuning
CHATML_SAMPLES ?= 50000
FINETUNE_DATA ?= finetune_data
FINETUNE_MODEL ?= QRK-300M
FINETUNE_ITERS ?= 1000
FINETUNE_BATCH ?= 2

prepare-chatml:
	pip install datasets
	mkdir -p $(FINETUNE_DATA)
	python prepare_chatml.py --samples $(CHATML_SAMPLES) --output $(FINETUNE_DATA)/train.jsonl

convert-model:
	python $(MLX_PRETRAIN)/convert-to-mlx-lm.py --run "$(FINETUNE_MODEL)" --out-path "$(FINETUNE_MODEL)-mlx"

finetune: check-finetune-data
	pip install "mlx-lm[train]"
	python -m mlx_lm.lora \
		--model $(FINETUNE_MODEL)-mlx \
		--data $(FINETUNE_DATA) \
		--train \
		--batch-size $(FINETUNE_BATCH) \
		--iters $(FINETUNE_ITERS)

fuse-lora:
	python -m mlx_lm.fuse \
		--model $(FINETUNE_MODEL)-mlx

check-finetune-data:
	@test -f $(FINETUNE_DATA)/train.jsonl || (echo "Error: Run 'make prepare-chatml' first" && exit 1)
	@test -d $(FINETUNE_MODEL)-mlx || (echo "Error: Run 'make convert-model' first" && exit 1)

test-tokenizer:
	@echo "Testing tokenizer..."
	python -c "from tokenizers import Tokenizer; t = Tokenizer.from_file('$(TOKENIZER_DIR)/tokenizer.json'); print('Vocab:', t.get_vocab_size()); r = t.encode('Hello <knowledge>test</knowledge>'); print('Tokens:', r.tokens)"
