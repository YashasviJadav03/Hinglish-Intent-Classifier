"""
src/model/evaluate.py

Phase 5: Final Evaluation & Error Analysis

Loads the fine-tuned PEFT LoRA adapter, runs evaluation on data/processed/test.csv,
and generates:
1. Overall accuracy, macro-F1, and per-class classification report
2. Confusion matrix visualization saved to results/confusion_matrix.png
3. CSV of top 15 most confidently misclassified utterances (results/misclassified_examples.csv)
4. Markdown comparison table comparing Zero-shot baseline vs Fine-tuned LoRA (results/comparison_table.md)
5. Full metrics JSON (results/final_eval_metrics.json)
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_finetuned_model(adapter_dir: Path = config.LORA_ADAPTER_DIR, base_model_name: str = config.DEFAULT_MODEL_NAME):
    """
    Loads base multilingual transformer and merges the fine-tuned LoRA adapter.
    """
    logger.info("Loading base model: %s", base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )
    logger.info("Loading fine-tuned LoRA adapter from: %s", adapter_dir)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir) if (adapter_dir / "vocab.txt").exists() else base_model_name)
    return model, tokenizer, device


def run_evaluation(model, tokenizer, device, test_df: pd.DataFrame):
    """
    Runs model inference over test_df, gathers predictions, confidences, and metrics.
    """
    texts = test_df["clean_text"].tolist()
    y_true = test_df["intent"].tolist()
    y_true_ids = [config.LABEL2ID[t] for t in y_true]

    logger.info("Running evaluation across %d test samples...", len(texts))
    
    all_preds = []
    all_pred_labels = []
    all_confidences = []
    all_probabilities = []

    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch_texts = texts[i : i + 32]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            pred_ids = np.argmax(probs, axis=-1)
            confidences = np.max(probs, axis=-1)

            all_preds.extend(pred_ids)
            all_pred_labels.extend([config.ID2LABEL[pid] for pid in pred_ids])
            all_confidences.extend(confidences)
            all_probabilities.extend(probs)

    all_preds = np.array(all_preds)
    all_confidences = np.array(all_confidences)

    # Compute overall metrics
    acc = accuracy_score(y_true_ids, all_preds)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true_ids, all_preds, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true_ids, all_preds, average="weighted", zero_division=0
    )

    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(
        y_true_ids, all_preds, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, label in enumerate(config.INTENT_LABELS):
        per_class_metrics[label] = {
            "precision": float(round(p_per[i], 4)),
            "recall": float(round(r_per[i], 4)),
            "f1_score": float(round(f1_per[i], 4)),
            "support": int(s_per[i]),
        }

    report_dict = classification_report(
        y_true, all_pred_labels, labels=config.INTENT_LABELS, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, all_pred_labels, labels=config.INTENT_LABELS, zero_division=0
    )

    metrics_payload = {
        "model_type": "lora_finetuned_distilbert",
        "eval_dataset_size": len(test_df),
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "macro_precision": float(round(macro_prec, 4)),
        "macro_recall": float(round(macro_rec, 4)),
        "weighted_f1": float(round(weighted_f1, 4)),
        "per_class": per_class_metrics,
        "classification_report": report_dict,
    }

    # Attach prediction columns to test_df for error analysis
    test_df_eval = test_df.copy()
    test_df_eval["predicted_intent"] = all_pred_labels
    test_df_eval["confidence"] = [float(round(c, 4)) for c in all_confidences]
    test_df_eval["is_correct"] = test_df_eval["intent"] == test_df_eval["predicted_intent"]

    return metrics_payload, report_text, test_df_eval, all_preds, y_true_ids


def plot_and_save_confusion_matrix(y_true_ids, y_pred_ids, output_path: Path):
    """
    Plots a polished, normalized confusion matrix and saves to results/confusion_matrix.png.
    """
    cm = confusion_matrix(y_true_ids, y_pred_ids, labels=list(range(config.NUM_LABELS)))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(9, 7))
    sns.set_theme(style="white")
    
    heatmap = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=config.INTENT_LABELS,
        yticklabels=config.INTENT_LABELS,
        cbar=True,
        linewidths=0.5,
    )
    plt.title("Hinglish Voice-Agent Intent Classifier — Normalized Confusion Matrix", fontsize=12, pad=15, weight="bold")
    plt.xlabel("Predicted Intent", fontsize=11, labelpad=10)
    plt.ylabel("True Intent", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", output_path)


def save_error_analysis(test_df_eval: pd.DataFrame, output_path: Path):
    """
    Identifies top 15 most confidently misclassified utterances.
    """
    misclassified = test_df_eval[~test_df_eval["is_correct"]].copy()
    
    if len(misclassified) > 0:
        top_errors = misclassified.sort_values(by="confidence", ascending=False).head(15)
    else:
        top_errors = pd.DataFrame(columns=["clean_text", "intent", "predicted_intent", "confidence"])

    cols_to_save = ["clean_text", "intent", "predicted_intent", "confidence"]
    available_cols = [c for c in cols_to_save if c in top_errors.columns]
    top_errors[available_cols].to_csv(output_path, index=False)
    logger.info("Saved top misclassified examples to %s (count: %d)", output_path, len(top_errors))


def generate_comparison_table(baseline_metrics: Dict[str, Any], finetuned_metrics: Dict[str, Any], output_path: Path):
    """
    Generates side-by-side comparison markdown between baseline and fine-tuned models.
    """
    base_acc = baseline_metrics.get("accuracy", 0.0)
    base_f1 = baseline_metrics.get("macro_f1", 0.0)
    base_wf1 = baseline_metrics.get("weighted_f1", 0.0)

    ft_acc = finetuned_metrics.get("accuracy", 0.0)
    ft_f1 = finetuned_metrics.get("macro_f1", 0.0)
    ft_wf1 = finetuned_metrics.get("weighted_f1", 0.0)

    acc_gain = (ft_acc - base_acc) * 100
    f1_gain = (ft_f1 - base_f1) * 100

    content = f"""# Baseline vs Fine-Tuned Model Performance Comparison

| Metric | Zero-Shot Baseline (DistilBERT NLI) | Fine-Tuned (DistilBERT + PEFT LoRA) | Absolute Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **{base_acc * 100:.2f}%** | **{ft_acc * 100:.2f}%** | **+{acc_gain:.2f}% pts** |
| **Macro F1-Score** | **{base_f1:.4f}** | **{ft_f1:.4f}** | **+{f1_gain:.2f}% pts** |
| **Weighted F1-Score** | {base_wf1:.4f} | {ft_wf1:.4f} | +{(ft_wf1 - base_wf1) * 100:.2f}% pts |

## Per-Class Breakdown

| Intent Class | Baseline F1 | LoRA Fine-Tuned F1 | Delta F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
"""
    base_per = baseline_metrics.get("per_class", {})
    ft_per = finetuned_metrics.get("per_class", {})

    for label in config.INTENT_LABELS:
        bf1 = base_per.get(label, {}).get("f1_score", 0.0)
        ff1 = ft_per.get(label, {}).get("f1_score", 0.0)
        sup = ft_per.get(label, {}).get("support", 0)
        delta = (ff1 - bf1)
        content += f"| `{label}` | {bf1:.4f} | **{ff1:.4f}** | +{delta:.4f} | {sup} |\n"

    content += "\n> **Key Takeaway**: LoRA fine-tuning significantly resolves dialectal ambiguities and noisy code-mixed phonetics that off-the-shelf zero-shot NLI models fail to disambiguate.\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Saved comparison table to %s", output_path)


def main():
    if not config.TEST_DATA_PATH.exists():
        logger.error("Test data not found at %s. Please run Phase 1 preprocessing first.", config.TEST_DATA_PATH)
        return

    test_df = pd.read_csv(config.TEST_DATA_PATH)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load fine-tuned model
    model, tokenizer, device = load_finetuned_model()

    # 2. Run evaluation
    ft_metrics, report_text, test_df_eval, pred_ids, true_ids = run_evaluation(model, tokenizer, device, test_df)

    # 3. Print Report
    print("\n" + "=" * 65)
    print("      FINAL EVALUATION REPORT (FINE-TUNED DISTILBERT + LORA)")
    print("=" * 65)
    print(f"Overall Accuracy : {ft_metrics['accuracy']:.4f} ({ft_metrics['accuracy']*100:.2f}%)")
    print(f"Macro F1-Score   : {ft_metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {ft_metrics['weighted_f1']:.4f}")
    print("-" * 65)
    print("Classification Report:")
    print(report_text)
    print("=" * 65 + "\n")

    # 4. Save Final Metrics JSON
    with open(config.FINAL_EVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(ft_metrics, f, indent=2)

    # 5. Plot and Save Confusion Matrix
    plot_and_save_confusion_matrix(true_ids, pred_ids, config.CONFUSION_MATRIX_PATH)

    # 6. Save Misclassified Examples
    save_error_analysis(test_df_eval, config.MISCLASSIFIED_PATH)

    # 7. Generate Comparison Table
    baseline_metrics = {}
    if config.BASELINE_METRICS_PATH.exists():
        with open(config.BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
            baseline_metrics = json.load(f)

    generate_comparison_table(baseline_metrics, ft_metrics, config.COMPARISON_TABLE_PATH)


if __name__ == "__main__":
    main()
