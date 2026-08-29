"""
Configuration settings, directory paths, and default hyperparameters
for the Hinglish Intent Classifier.
"""

import os
from pathlib import Path
from typing import Dict, List

# ==========================================
# Base & Directory Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = BASE_DIR / "models"
LORA_ADAPTER_DIR = MODELS_DIR / "lora-adapter"
BEST_MODEL_DIR = MODELS_DIR / "best-model"

RESULTS_DIR = BASE_DIR / "results"
EXPERIMENT_LOG_PATH = RESULTS_DIR / "experiment_log.csv"
BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
FINAL_EVAL_METRICS_PATH = RESULTS_DIR / "final_eval_metrics.json"
MISCLASSIFIED_PATH = RESULTS_DIR / "misclassified_examples.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
COMPARISON_TABLE_PATH = RESULTS_DIR / "comparison_table.md"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SRC_DIR = BASE_DIR / "src"

# ==========================================
# Intent Classes & Label Mappings
# ==========================================
INTENT_LABELS: List[str] = [
    "complaint",
    "purchase_inquiry",
    "price_negotiation",
    "callback_request",
    "not_interested",
    "positive_confirmation",
]

ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(INTENT_LABELS)}
LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(INTENT_LABELS)}
NUM_LABELS: int = len(INTENT_LABELS)

# ==========================================
# Model & Preprocessing Hyperparameters
# ==========================================
DEFAULT_MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LENGTH = 128
SEED = 42

# Training Hyperparameters
DEFAULT_TRAIN_PARAMS = {
    "model_name_or_path": DEFAULT_MODEL_NAME,
    "num_train_epochs": 5,
    "learning_rate": 3e-4,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "gradient_accumulation_steps": 1,
    "logging_steps": 20,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "macro_f1",
}

# LoRA Adapter Hyperparameters
DEFAULT_LORA_PARAMS = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["q_lin", "v_lin"],  # Standard for DistilBERT attention layers
    "bias": "none",
    "task_type": "SEQ_CLS",
}

# ==========================================
# API Service Configurations
# ==========================================
API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 7860))  # HF Spaces uses 7860
API_TITLE = "Hinglish Voice-Agent Intent Classification API"
API_VERSION = "1.0.0"
