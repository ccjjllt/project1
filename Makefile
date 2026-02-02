.PHONY: help setup build-data train-tfidf train-lstm train-attn ablation error-analysis

help:
	@echo "Targets:"
	@echo "  setup          Install deps into current env"
	@echo "  build-data     Clean & split CSV dataset"
	@echo "  train-tfidf    Train TF-IDF + LR baseline"
	@echo "  train-lstm     Train BiLSTM pooling classifier"
	@echo "  train-attn     Train BiLSTM + Attention classifier"
	@echo "  ablation       Run ablation (pooling/attention/etc.)"
	@echo "  error-analysis Run error analysis on best checkpoint"

setup:
	pip install -r requirements.txt
	pip install -e .

build-data:
	python scripts/build_dataset.py --config configs/dataset_csv.yaml

train-tfidf:
	python scripts/train_tfidf.py --config configs/tfidf.yaml

train-lstm:
	python scripts/train_lstm.py --config configs/lstm.yaml

train-attn:
	python scripts/train_lstm_attn.py --config configs/lstm_attn.yaml

ablation:
	python scripts/run_ablation.py --config configs/ablation.yaml

error-analysis:
	python scripts/run_error_analysis.py --config configs/error_analysis.yaml
