"""
src/model/baseline_eval.py

Evaluates classical and zero-shot baselines on the leakage-free Hinglish dataset
(data/processed/train.csv, test.csv, test_ood.csv).

Evaluates:
1. TF-IDF + Multinomial Naive Bayes
2. TF-IDF + Logistic Regression
3. TF-IDF + Linear Support Vector Classifier (LinearSVC)
4. TF-IDF + Random Forest Classifier
5. Zero-Shot Multilingual / Semantic Keyword Baseline

Computes:
- Overall Accuracy
- Macro-F1 & Weighted-F1
- Per-class Precision, Recall, F1, and Support
- In-domain (test.csv) and Out-of-Distribution (test_ood.csv) performance
Saves metrics to results/baseline_metrics.json and prints a formatted classification report.
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model_pipeline(
    name: str,
    pipeline_obj: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ood_df: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Fits a scikit-learn pipeline on train_df and evaluates on test_df and optionally ood_df.
    """
    logger.info("Training classical baseline: %s...", name)
    pipeline_obj.fit(train_df["clean_text"], train_df["intent"])

    # In-Domain Test Predictions
    test_preds = pipeline_obj.predict(test_df["clean_text"])
    y_test = test_df["intent"].tolist()

    acc = accuracy_score(y_test, test_preds)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_test, test_preds, labels=config.INTENT_LABELS, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_test, test_preds, labels=config.INTENT_LABELS, average="weighted", zero_division=0
    )
    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(
        y_test, test_preds, labels=config.INTENT_LABELS, average=None, zero_division=0
    )

    per_class = {}
    for i, label in enumerate(config.INTENT_LABELS):
        per_class[label] = {
            "precision": float(round(p_per[i], 4)),
            "recall": float(round(r_per[i], 4)),
            "f1_score": float(round(f1_per[i], 4)),
            "support": int(s_per[i]),
        }

    report_dict = classification_report(
        y_test, test_preds, labels=config.INTENT_LABELS, output_dict=True, zero_division=0
    )

    result = {
        "model_name": name,
        "test_accuracy": float(round(acc, 4)),
        "test_macro_f1": float(round(macro_f1, 4)),
        "test_macro_precision": float(round(macro_prec, 4)),
        "test_macro_recall": float(round(macro_rec, 4)),
        "test_weighted_f1": float(round(weighted_f1, 4)),
        "per_class": per_class,
        "classification_report": report_dict,
    }

    if ood_df is not None:
        ood_preds = pipeline_obj.predict(ood_df["clean_text"])
        y_ood = ood_df["intent"].tolist()
        ood_acc = accuracy_score(y_ood, ood_preds)
        ood_prec, ood_rec, ood_f1, _ = precision_recall_fscore_support(
            y_ood, ood_preds, labels=config.INTENT_LABELS, average="macro", zero_division=0
        )
        result["ood_accuracy"] = float(round(ood_acc, 4))
        result["ood_macro_f1"] = float(round(ood_f1, 4))

    return result


def run_zero_shot_baseline(test_df: pd.DataFrame, ood_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Evaluates a semantic keyword / heuristic zero-shot baseline.
    """
    keywords = {
        "complaint": ["kharab", "bekar", "refund", "complain", "delay", "issue", "defective", "cheat", "fraud", "fault", "broken", "deliver nahi", "stale", "kam nahi"],
        "purchase_inquiry": ["detail", "brochure", "information", "available", "specification", "mileage", "warranty", "feature", "emi", "kya", "price", "kitna", "stock", "color"],
        "price_negotiation": ["discount", "kam", "sasta", "cheaper", "negotiate", "offer", "expensive", "coupon", "bargain", "rate", "off", "budget", "cut", "kam karo"],
        "callback_request": ["call", "baad", "busy", "meeting", "driving", "kal", "sham", "schedule", "ring", "later", "phone", "ghante", "morning", "connect"],
        "not_interested": ["nahi chahiye", "not interested", "dnd", "mat call", "stop", "unsubscribe", "block", "remove", "don't", "no thanks", "mana", "close my lead"],
        "positive_confirmation": ["haan", "yes", "confirm", "done", "agree", "theek", "ready", "proceed", "pack", "lock", "approved", "seal", "final", "gpay"],
    }

    def predict_zero_shot(text: str) -> str:
        t_lower = text.lower()
        scores = {k: sum(1 for kw in kw_list if kw in t_lower) for k, kw_list in keywords.items()}
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            best_intent = "purchase_inquiry"
        return best_intent

    y_test = test_df["intent"].tolist()
    test_preds = [predict_zero_shot(t) for t in test_df["clean_text"]]

    acc = accuracy_score(y_test, test_preds)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_test, test_preds, labels=config.INTENT_LABELS, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_test, test_preds, labels=config.INTENT_LABELS, average="weighted", zero_division=0
    )
    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(
        y_test, test_preds, labels=config.INTENT_LABELS, average=None, zero_division=0
    )

    per_class = {}
    for i, label in enumerate(config.INTENT_LABELS):
        per_class[label] = {
            "precision": float(round(p_per[i], 4)),
            "recall": float(round(r_per[i], 4)),
            "f1_score": float(round(f1_per[i], 4)),
            "support": int(s_per[i]),
        }

    report_dict = classification_report(
        y_test, test_preds, labels=config.INTENT_LABELS, output_dict=True, zero_division=0
    )

    result = {
        "model_name": "Zero-Shot Heuristic Keyword Baseline",
        "test_accuracy": float(round(acc, 4)),
        "test_macro_f1": float(round(macro_f1, 4)),
        "test_macro_precision": float(round(macro_prec, 4)),
        "test_macro_recall": float(round(macro_rec, 4)),
        "test_weighted_f1": float(round(weighted_f1, 4)),
        "per_class": per_class,
        "classification_report": report_dict,
    }

    if ood_df is not None:
        y_ood = ood_df["intent"].tolist()
        ood_preds = [predict_zero_shot(t) for t in ood_df["clean_text"]]
        ood_acc = accuracy_score(y_ood, ood_preds)
        ood_prec, ood_rec, ood_f1, _ = precision_recall_fscore_support(
            y_ood, ood_preds, labels=config.INTENT_LABELS, average="macro", zero_division=0
        )
        result["ood_accuracy"] = float(round(ood_acc, 4))
        result["ood_macro_f1"] = float(round(ood_f1, 4))

    return result


def main():
    if not config.TRAIN_DATA_PATH.exists() or not config.TEST_DATA_PATH.exists():
        logger.error("Dataset not found. Please run preprocess.py first.")
        return

    logger.info("Loading training and testing datasets...")
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    ood_df = pd.read_csv(config.TEST_OOD_DATA_PATH) if config.TEST_OOD_DATA_PATH.exists() else None

    logger.info("Train samples: %d | In-Domain Test samples: %d | OOD Test samples: %d",
                len(train_df), len(test_df), len(ood_df) if ood_df is not None else 0)

    # Define TF-IDF Classical Pipelines
    pipelines = {
        "TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=config.SEED, C=1.0)),
        ]),
        "TF-IDF + Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("clf", LinearSVC(random_state=config.SEED, C=1.0)),
        ]),
        "TF-IDF + Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
        "TF-IDF + Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=config.SEED, n_jobs=-1)),
        ]),
    }

    all_metrics = {}
    summary_rows = []

    # Run Classical Baselines
    for name, pipe in pipelines.items():
        res = evaluate_model_pipeline(name, pipe, train_df, test_df, ood_df=ood_df)
        all_metrics[name] = res
        summary_rows.append({
            "Model": name,
            "Test Accuracy": f"{res['test_accuracy']*100:.1f}%",
            "Test Macro F1": f"{res['test_macro_f1']:.4f}",
            "OOD Accuracy": f"{res.get('ood_accuracy', 0)*100:.1f}%" if ood_df is not None else "N/A",
            "OOD Macro F1": f"{res.get('ood_macro_f1', 0):.4f}" if ood_df is not None else "N/A",
        })

    # Run Zero-Shot Baseline
    zs_res = run_zero_shot_baseline(test_df, ood_df=ood_df)
    all_metrics["Zero-Shot Baseline"] = zs_res
    summary_rows.append({
        "Model": "Zero-Shot Heuristic Keyword Baseline",
        "Test Accuracy": f"{zs_res['test_accuracy']*100:.1f}%",
        "Test Macro F1": f"{zs_res['test_macro_f1']:.4f}",
        "OOD Accuracy": f"{zs_res.get('ood_accuracy', 0)*100:.1f}%" if ood_df is not None else "N/A",
        "OOD Macro F1": f"{zs_res.get('ood_macro_f1', 0):.4f}" if ood_df is not None else "N/A",
    })

    # Print baseline comparison table
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("                    CLASSICAL & ZERO-SHOT BASELINE BENCHMARK")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    # Save to results/baseline_metrics.json
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BASELINE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Baseline metrics saved to %s", config.BASELINE_METRICS_PATH)


if __name__ == "__main__":
    main()
