"""
src/model/train.py

Fine-tunes a multilingual transformer (distilbert-base-multilingual-cased)
for sequence classification using HuggingFace Transformers and PEFT LoRA adapters.

Supports CLI arguments for hyperparameter exploration and ablation studies
(learning rate, LoRA rank, epochs, batch size), logs validation metrics to
results/experiment_log.csv, and saves the trained adapter to models/lora-adapter/.
"""

import os
import sys
import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HinglishIntentDataset(Dataset):
    """
    PyTorch Dataset for tokenized Hinglish text classification.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamically padded by collator
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(eval_pred):
    """
    Computes accuracy and macro-F1 for evaluation during training.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "macro_precision": float(round(precision, 4)),
        "macro_recall": float(round(recall, 4)),
    }


def log_experiment_result(run_info: Dict[str, Any]):
    """
    Appends experiment hyperparameter and metric results to results/experiment_log.csv.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.EXPERIMENT_LOG_PATH
    
    df_new = pd.DataFrame([run_info])
    if log_file.exists():
        df_existing = pd.read_csv(log_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(log_file, index=False)
    logger.info("Logged experiment run to %s", log_file)


def train_lora_model(
    model_name: str = config.DEFAULT_MODEL_NAME,
    learning_rate: float = 3e-4,
    num_train_epochs: int = 5,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    output_dir: Path = config.LORA_ADAPTER_DIR,
    run_name: str = None,
) -> Dict[str, Any]:
    """
    Sets up PEFT LoRA adapter on base model, runs training loop, and saves model artifacts.
    """
    if run_name is None:
        run_name = f"lora_r{lora_r}_lr{learning_rate}_ep{num_train_epochs}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("Starting run: %s", run_name)
    logger.info("Base Model: %s | LoRA Rank: %d | LoRA Alpha: %d | LR: %e | Epochs: %d",
                model_name, lora_r, lora_alpha, learning_rate, num_train_epochs)

    # 1. Load Data
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VAL_DATA_PATH)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = HinglishIntentDataset(
        train_df["clean_text"].tolist(), train_df["label"].tolist(), tokenizer, config.MAX_LENGTH
    )
    val_dataset = HinglishIntentDataset(
        val_df["clean_text"].tolist(), val_df["label"].tolist(), tokenizer, config.MAX_LENGTH
    )

    # 2. Base Model Setup
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    # 3. LoRA Configuration
    # Automatically identify target modules (query & value projection)
    target_modules = ["q_lin", "v_lin"] if "distilbert" in model_name.lower() else ["query", "value"]
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 4. Training Arguments
    run_output_dir = output_dir if str(output_dir).endswith(run_name) else output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_output_dir / "checkpoints"),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        seed=config.SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # 5. Execute Training
    train_result = trainer.train()
    logger.info("Training complete. Evaluating on validation set...")

    eval_metrics = trainer.evaluate()
    logger.info("Validation Results: %s", eval_metrics)

    # 6. Save Adapter & Tokenizer
    model.save_pretrained(str(run_output_dir))
    tokenizer.save_pretrained(str(run_output_dir))
    
    # Also save/link as latest default adapter
    config.LORA_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.LORA_ADAPTER_DIR))
    tokenizer.save_pretrained(str(config.LORA_ADAPTER_DIR))

    logger.info("Saved best adapter to %s and %s", run_output_dir, config.LORA_ADAPTER_DIR)

    # 7. Record Experiment Log
    run_record = {
        "run_name": run_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_name": model_name,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "epochs": num_train_epochs,
        "train_batch_size": train_batch_size,
        "train_loss": float(round(train_result.training_loss, 4)),
        "val_loss": float(round(eval_metrics.get("eval_loss", 0.0), 4)),
        "val_accuracy": float(round(eval_metrics.get("eval_accuracy", 0.0), 4)),
        "val_macro_f1": float(round(eval_metrics.get("eval_macro_f1", 0.0), 4)),
        "adapter_dir": str(run_output_dir),
    }

    log_experiment_result(run_record)
    return run_record


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Hinglish Intent Classifier with LoRA")
    parser.add_argument("--model_name", type=str, default=config.DEFAULT_MODEL_NAME, help="Base HF model")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--run_name", type=str, default=None, help="Name for experiment run")
    return parser.parse_args()


def main():
    args = parse_args()
    train_lora_model(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
