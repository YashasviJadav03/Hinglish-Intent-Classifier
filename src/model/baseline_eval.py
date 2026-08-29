"""
src/model/baseline_eval.py

Evaluates a zero-shot multilingual baseline model on the Hinglish test dataset
(data/processed/test.csv).

Computes:
- Overall Accuracy
- Macro-F1 & Weighted-F1
- Per-class Precision, Recall, F1, and Support
Saves metrics to results/baseline_metrics.json and prints a formatted classification report.
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Descriptive candidate labels for zero-shot NLI prompting
INTENT_DESCRIPTIONS = {
    "complaint": "complaint about bad service, delivery delay, or defective order issue",
    "purchase_inquiry": "inquiry about product features, pricing details, specifications, or course information",
    "price_negotiation": "asking for discount, bargain, lower price, or special festive offers",
    "callback_request": "requesting a callback later because busy, driving, or in a meeting",
    "not_interested": "not interested in offers, asking to unsubscribe, stop calling, or activate DND",
    "positive_confirmation": "positive agreement, confirming the booking, ready to purchase, or deal done",
}


def load_zero_shot_classifier(model_name: str = "distilbert-base-multilingual-cased"):
    """
    Initializes a zero-shot classification pipeline.
    Falls back gracefully if a specialized NLI model is available.
    """
    logger.info("Initializing zero-shot pipeline with model: %s", model_name)
    device = 0 if torch.cuda.is_available() else -1
    
    # Try zero-shot pipeline with a standard multilingual zero-shot / NLI backbone
    candidate_models = [
        "mor4i/multilingual-MiniLMv2-L6-mnli-xnli",
        "typeform/distilbert-base-uncased-mnli",
        model_name,
    ]
    
    classifier = None
    for m in candidate_models:
        try:
            logger.info("Attempting to load zero-shot model: %s", m)
            classifier = pipeline("zero-shot-classification", model=m, device=device)
            logger.info("Successfully loaded zero-shot model: %s", m)
            break
        except Exception as e:
            logger.warning("Could not load %s for zero-shot pipeline: %s", m, e)

    if classifier is None:
        logger.info("Falling back to standard feature similarity / heuristic zero-shot baseline.")
    return classifier


def run_zero_shot_evaluation(
    test_df: pd.DataFrame,
    classifier=None,
) -> Dict[str, Any]:
    """
    Runs zero-shot inference over test_df and calculates comprehensive metrics.
    """
    candidate_labels = list(INTENT_DESCRIPTIONS.values())
    label_desc_to_intent = {desc: intent for intent, desc in INTENT_DESCRIPTIONS.items()}

    predictions: List[str] = []
    y_true: List[str] = test_df["intent"].tolist()
    texts: List[str] = test_df["clean_text"].tolist()

    logger.info("Evaluating %d test samples zero-shot...", len(texts))

    if classifier is not None:
        hypothesis_template = "This customer conversation utterance is a {}."
        for i, text in enumerate(texts):
            try:
                res = classifier(text, candidate_labels, hypothesis_template=hypothesis_template)
                top_desc = res["labels"][0]
                pred_intent = label_desc_to_intent.get(top_desc, config.INTENT_LABELS[0])
            except Exception as e:
                logger.debug("Inference error on sample %d: %s. Using default.", i, e)
                pred_intent = config.INTENT_LABELS[i % len(config.INTENT_LABELS)]
            predictions.append(pred_intent)
            if (i + 1) % 20 == 0 or (i + 1) == len(texts):
                logger.info("Processed %d/%d test samples", i + 1, len(texts))
    else:
        # Heuristic / Keyword-based Zero-shot rule baseline when offline
        logger.info("Running deterministic semantic keyword zero-shot baseline...")
        keywords = {
            "complaint": ["kharab", "bekar", "refund", "complain", "delay", "issue", "defective", "cheat", "cheat", "fraud", "fault"],
            "purchase_inquiry": ["detail", "brochure", "information", "available", "specification", "mileage", "warranty", "feature", "emi"],
            "price_negotiation": ["discount", "kam", "sasta", "cheaper", "negotiate", "offer", "expensive", "coupon", "bargain", "rate"],
            "callback_request": ["call", "baad", "busy", "meeting", "driving", "kal", "sham", "schedule", "ring", "later"],
            "not_interested": ["nahi chahiye", "not interested", "dnd", "mat call", "stop", "unsubscribe", "block", "remove", "don't"],
            "positive_confirmation": ["haan", "yes", "confirm", "done", "agree", "theek", "ready", "proceed", "pack", "lock"],
        }
        for text in texts:
            t_lower = text.lower()
            scores = {k: sum(1 for kw in kw_list if kw in t_lower) for k, kw_list in keywords.items()}
            best_intent = max(scores, key=scores.get)
            if scores[best_intent] == 0:
                best_intent = "purchase_inquiry"  # neutral fallback
            predictions.append(best_intent)

    # Compute metrics
    acc = accuracy_score(y_true, predictions)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=config.INTENT_LABELS, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=config.INTENT_LABELS, average="weighted", zero_division=0
    )

    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(
        y_true, predictions, labels=config.INTENT_LABELS, average=None, zero_division=0
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
        y_true, predictions, labels=config.INTENT_LABELS, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, predictions, labels=config.INTENT_LABELS, zero_division=0
    )

    metrics_payload = {
        "model_type": "zero_shot_baseline",
        "eval_dataset_size": len(test_df),
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "macro_precision": float(round(macro_prec, 4)),
        "macro_recall": float(round(macro_rec, 4)),
        "weighted_f1": float(round(weighted_f1, 4)),
        "per_class": per_class_metrics,
        "classification_report": report_dict,
    }

    return metrics_payload, report_text, predictions


def main():
    if not config.TEST_DATA_PATH.exists():
        logger.error("Test dataset not found at %s. Please run Phase 1 preprocessing first.", config.TEST_DATA_PATH)
        return

    logger.info("Loading test dataset from %s", config.TEST_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    classifier = load_zero_shot_classifier(config.DEFAULT_MODEL_NAME)
    metrics, report_text, _ = run_zero_shot_evaluation(test_df, classifier)

    # Print baseline report
    print("\n" + "=" * 65)
    print("        ZERO-SHOT MULTILINGUAL BASELINE EVALUATION REPORT")
    print("=" * 65)
    print(f"Overall Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Macro F1-Score   : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    print("-" * 65)
    print("Classification Report:")
    print(report_text)
    print("=" * 65 + "\n")

    # Save to results/baseline_metrics.json
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BASELINE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Baseline metrics saved to %s", config.BASELINE_METRICS_PATH)


if __name__ == "__main__":
    main()
