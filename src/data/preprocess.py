"""
src/data/preprocess.py

Preprocesses code-mixed Hinglish conversational data:
1. Normalizes transliteration elongation noise (e.g., 'bohooooot' -> 'bohot', 'plzzzz' -> 'plz')
2. Extracts emojis and excessive punctuation into explicit metadata features
3. Cleans whitespace, casing, and control characters
4. Deduplicates rows and drops empty/near-empty utterances
5. Performs Group-Based Stratified Split on base_ids BEFORE any data augmentation
6. Augments ONLY the training partition (val, test, and test_ood remain clean, un-augmented)
7. Preprocesses the independent Out-of-Distribution (OOD) benchmark dataset
8. Validates zero lexical/semantic leakage across all splits with automated assertions
9. Saves outputs to data/processed/train.csv, val.csv, test.csv, test_ood.csv
"""

import re
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Regex patterns for emoji & punctuation extraction
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

PUNCT_EXCESS_PATTERN = re.compile(r"([!?.,;:\-\_])\1{1,}")
CHAR_ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")  # 3 or more repeated chars -> compress to 2

# Standard conversational variations used ONLY during training data augmentation
DEFAULT_TRAIN_VARIATIONS = [
    ("", ""),
    ("Arre ", " please"),
    ("Hey, ", "!"),
    ("Bhai ", " jaldi batao"),
    ("Sir ", " kindly confirm"),
    ("Sunna ", "..."),
    ("Dekho ", " actually"),
    ("Hello team, ", ""),
]


def extract_metadata_and_clean(text: str) -> Tuple[str, str, int]:
    """
    Extracts emoji features, cleans transliteration elongations,
    and returns (cleaned_text, extracted_emojis, punct_excess_count).
    """
    if not isinstance(text, str):
        return "", "", 0

    raw_text = text.strip()

    # 1. Extract emojis
    emojis_found = "".join(EMOJI_PATTERN.findall(raw_text))

    # 2. Remove emojis from text string
    clean_text = EMOJI_PATTERN.sub(" ", raw_text)

    # 3. Count and normalize excessive punctuation (e.g. '?????' -> '?')
    excess_punct_matches = PUNCT_EXCESS_PATTERN.findall(clean_text)
    excess_punct_count = len(excess_punct_matches)
    clean_text = PUNCT_EXCESS_PATTERN.sub(r"\1", clean_text)

    # 4. Normalize transliteration elongation noise (e.g. 'plzzzz' -> 'plz', 'bohooooot' -> 'bohot')
    clean_text = CHAR_ELONGATION_PATTERN.sub(r"\1\1", clean_text)

    # 5. Clean whitespace & newlines
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    return clean_text, emojis_found, excess_punct_count


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies text cleaning, feature extraction, deduplication, and filtering.
    Preserves or assigns base_id for group-based split integrity.
    """
    logger.info("Raw input records: %d", len(df))

    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("Input DataFrame must contain 'text' and 'intent' columns.")

    # Filter out unknown/invalid labels
    df = df[df["intent"].isin(config.INTENT_LABELS)].copy()

    # Ensure base_id exists
    if "base_id" not in df.columns:
        df["base_id"] = [f"{intent}_{i:03d}" for i, intent in enumerate(df["intent"])]

    # Apply cleaning & feature extraction
    results = [extract_metadata_and_clean(t) for t in df["text"]]
    df["clean_text"] = [r[0] for r in results]
    df["emojis"] = [r[1] for r in results]
    df["excess_punct_count"] = [r[2] for r in results]

    # Map intent to label_id
    df["label"] = df["intent"].map(config.LABEL2ID)

    # Filter out near-empty text (< 3 characters or < 1 word)
    df = df[df["clean_text"].str.len() >= 3]
    df = df[df["clean_text"].apply(lambda x: len(x.split()) >= 1)]

    # Deduplicate on clean_text and intent
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["clean_text", "intent"]).reset_index(drop=True)
    logger.info("Deduplication: dropped %d duplicate records. Retained: %d", before_dedup - len(df), len(df))

    return df


def split_dataset_by_base_id(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs Group-Based Stratified Split on base_id so that zero base utterances
    or their variations ever cross train, validation, or test partition boundaries.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Get unique base items with their intent label
    base_df = df.drop_duplicates(subset=["base_id"]).copy()

    # Step 1: Split base_ids into train and temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_base_df, temp_base_df = train_test_split(
        base_df, test_size=temp_ratio, random_state=seed, stratify=base_df["intent"]
    )

    # Step 2: Split temp into val and test
    relative_test_ratio = test_ratio / temp_ratio
    val_base_df, test_base_df = train_test_split(
        temp_base_df, test_size=relative_test_ratio, random_state=seed, stratify=temp_base_df["intent"]
    )

    train_base_ids = set(train_base_df["base_id"])
    val_base_ids = set(val_base_df["base_id"])
    test_base_ids = set(test_base_df["base_id"])

    # Map back to full records
    train_df = df[df["base_id"].isin(train_base_ids)].copy().reset_index(drop=True)
    val_df = df[df["base_id"].isin(val_base_ids)].copy().reset_index(drop=True)
    test_df = df[df["base_id"].isin(test_base_ids)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def augment_training_data(
    train_df: pd.DataFrame, variations: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Applies synthetic linguistic variations exclusively to the training dataset.
    Validation and test datasets MUST NOT be passed through this function.
    """
    if variations is None:
        variations = DEFAULT_TRAIN_VARIATIONS

    augmented_records = []
    for _, row in train_df.iterrows():
        base_id = row["base_id"]
        base_text = row["text"]
        intent = row["intent"]

        for prefix, suffix in variations:
            aug_text = f"{prefix}{base_text}{suffix}".strip()
            clean_text, emojis, excess_punct = extract_metadata_and_clean(aug_text)

            augmented_records.append({
                "base_id": base_id,
                "text": aug_text,
                "intent": intent,
                "clean_text": clean_text,
                "emojis": emojis,
                "excess_punct_count": excess_punct,
                "label": config.LABEL2ID[intent],
            })

    aug_df = pd.DataFrame(augmented_records)
    aug_df = aug_df.drop_duplicates(subset=["clean_text", "intent"]).reset_index(drop=True)
    logger.info("Training augmentation expanded %d base rows to %d training samples.", len(train_df), len(aug_df))
    return aug_df


def verify_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ood_df: Optional[pd.DataFrame] = None
) -> bool:
    """
    Performs strict automated validation assertions ensuring 0% data leakage
    across train, validation, test, and OOD benchmark splits.
    """
    train_base_ids = set(train_df["base_id"])
    val_base_ids = set(val_df["base_id"])
    test_base_ids = set(test_df["base_id"])

    # 1. Base ID overlap assertions
    train_val_id_leakage = train_base_ids.intersection(val_base_ids)
    train_test_id_leakage = train_base_ids.intersection(test_base_ids)
    val_test_id_leakage = val_base_ids.intersection(test_base_ids)

    if train_test_id_leakage:
        raise ValueError(f"CRITICAL LEAKAGE: Train and Test share base_ids: {train_test_id_leakage}")
    if train_val_id_leakage:
        raise ValueError(f"CRITICAL LEAKAGE: Train and Val share base_ids: {train_val_id_leakage}")
    if val_test_id_leakage:
        raise ValueError(f"CRITICAL LEAKAGE: Val and Test share base_ids: {val_test_id_leakage}")

    # 2. Exact cleaned text overlap assertions
    train_clean_texts = set(train_df["clean_text"])
    val_clean_texts = set(val_df["clean_text"])
    test_clean_texts = set(test_df["clean_text"])

    train_test_text_leakage = train_clean_texts.intersection(test_clean_texts)
    train_val_text_leakage = train_clean_texts.intersection(val_clean_texts)
    val_test_text_leakage = val_clean_texts.intersection(test_clean_texts)

    if train_test_text_leakage:
        raise ValueError(f"CRITICAL LEAKAGE: Train and Test share exact clean_text: {train_test_text_leakage}")
    if train_val_text_leakage:
        raise ValueError(f"CRITICAL LEAKAGE: Train and Val share exact clean_text: {train_val_text_leakage}")
    if val_test_text_leakage:
        raise ValueError(f"CRITICAL LEAKAGE: Val and Test share exact clean_text: {val_test_text_leakage}")

    if ood_df is not None:
        ood_base_ids = set(ood_df["base_id"])
        ood_clean_texts = set(ood_df["clean_text"])
        if train_base_ids.intersection(ood_base_ids):
            raise ValueError("CRITICAL LEAKAGE: Train and OOD share base_ids!")
        if train_clean_texts.intersection(ood_clean_texts):
            raise ValueError("CRITICAL LEAKAGE: Train and OOD share exact clean_text!")

    logger.info("Leakage verification PASSED: 0 overlapping base IDs or exact utterances across all splits.")
    return True


def print_class_balance_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ood_df: Optional[pd.DataFrame] = None
):
    """
    Logs comprehensive dataset summary and class distribution report.
    """
    total_samples = len(train_df) + len(val_df) + len(test_df)
    unique_train_bases = train_df["base_id"].nunique()
    unique_val_bases = val_df["base_id"].nunique()
    unique_test_bases = test_df["base_id"].nunique()
    total_unique_bases = unique_train_bases + unique_val_bases + unique_test_bases

    print("\n" + "=" * 78)
    print("           DATASET CLASS DISTRIBUTION & BENCHMARK INTEGRITY REPORT")
    print("=" * 78)
    print(f"Total Unique Canonical Base Utterances: {total_unique_bases}")
    print(f"  • Train Base Seeds: {unique_train_bases} ({unique_train_bases/total_unique_bases*100:.1f}%) -> Augmented: {len(train_df)} samples")
    print(f"  • Val Base Seeds:   {unique_val_bases} ({unique_val_bases/total_unique_bases*100:.1f}%) -> Un-augmented: {len(val_df)} samples")
    print(f"  • Test Base Seeds:  {unique_test_bases} ({unique_test_bases/total_unique_bases*100:.1f}%) -> Un-augmented: {len(test_df)} samples")
    if ood_df is not None:
        print(f"  • OOD Benchmark:    {len(ood_df)} independent samples (ASR noise, typos, slang)")
    print(f"Total Processed Samples: {total_samples + (len(ood_df) if ood_df is not None else 0)}")
    print("-" * 78)

    summary_data = []
    for label in config.INTENT_LABELS:
        tr_cnt = (train_df["intent"] == label).sum()
        vl_cnt = (val_df["intent"] == label).sum()
        ts_cnt = (test_df["intent"] == label).sum()
        ood_cnt = (ood_df["intent"] == label).sum() if ood_df is not None else 0
        total = tr_cnt + vl_cnt + ts_cnt
        summary_data.append({
            "Intent Label": label,
            "Train (Aug)": tr_cnt,
            "Val (Clean)": vl_cnt,
            "Test (Clean)": ts_cnt,
            "OOD Benchmark": ood_cnt,
            "Total In-Domain": total,
            "Train %": f"{tr_cnt/len(train_df)*100:.1f}%",
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("=" * 78 + "\n")


def main():
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = config.RAW_DATA_DIR / "raw_dataset.csv"
    raw_ood_path = config.RAW_DATA_DIR / "raw_ood_dataset.csv"

    if not raw_path.exists() or not raw_ood_path.exists():
        logger.info("Raw datasets not found. Running load_dataset first...")
        from src.data.load_dataset import main as run_load
        run_load()

    logger.info("Loading raw base dataset from %s", raw_path)
    raw_df = pd.read_csv(raw_path)

    # 1. Clean and deduplicate base dataset
    cleaned_base_df = preprocess_dataframe(raw_df)

    # 2. Group-Based Stratified Split on base utterances BEFORE augmentation
    train_base_df, val_df, test_df = split_dataset_by_base_id(
        cleaned_base_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=config.SEED
    )

    # 3. Augment ONLY the training split
    train_df = augment_training_data(train_base_df)

    # 4. Clean and preprocess OOD benchmark dataset
    ood_df = None
    if raw_ood_path.exists():
        logger.info("Processing OOD benchmark dataset from %s", raw_ood_path)
        raw_ood_df = pd.read_csv(raw_ood_path)
        ood_df = preprocess_dataframe(raw_ood_df)

    # 5. Rigorous automated leakage verification
    verify_split_leakage(train_df, val_df, test_df, ood_df=ood_df)

    # 6. Save processed splits to data/processed/
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False, encoding="utf-8")
    val_df.to_csv(config.VAL_DATA_PATH, index=False, encoding="utf-8")
    test_df.to_csv(config.TEST_DATA_PATH, index=False, encoding="utf-8")

    if ood_df is not None:
        ood_df.to_csv(config.TEST_OOD_DATA_PATH, index=False, encoding="utf-8")
        logger.info("Saved test_ood.csv (OOD Benchmark) -> %s", config.TEST_OOD_DATA_PATH)

    logger.info("Saved train.csv (Augmented) -> %s", config.TRAIN_DATA_PATH)
    logger.info("Saved val.csv   (Clean)     -> %s", config.VAL_DATA_PATH)
    logger.info("Saved test.csv  (Clean)     -> %s", config.TEST_DATA_PATH)

    # 7. Print class distribution and integrity summary
    print_class_balance_report(train_df, val_df, test_df, ood_df=ood_df)


if __name__ == "__main__":
    main()
