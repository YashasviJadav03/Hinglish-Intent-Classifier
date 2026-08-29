"""
src/data/preprocess.py

Preprocesses code-mixed Hinglish conversational data:
1. Normalizes transliteration elongation noise (e.g., 'bohooooot' -> 'bohot', 'plzzzz' -> 'plz')
2. Extracts emojis and excessive punctuation into explicit metadata features
3. Cleans whitespace, casing, and control characters
4. Deduplicates rows and drops empty/near-empty utterances
5. Performs stratified 70/15/15 train/val/test split
6. Saves outputs to data/processed/train.csv, val.csv, test.csv
7. Prints detailed class balance and summary metrics
"""

import re
import sys
import logging
from pathlib import Path
from typing import Tuple

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
    """
    logger.info("Raw input records: %d", len(df))

    # Ensure required columns
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("Input DataFrame must contain 'text' and 'intent' columns.")

    # Filter out unknown/invalid labels
    df = df[df["intent"].isin(config.INTENT_LABELS)].copy()

    # Apply cleaning & feature extraction
    results = [extract_metadata_and_clean(t) for t in df["text"]]
    df["clean_text"] = [r[0] for r in results]
    df["emojis"] = [r[1] for r in results]
    df["excess_punct_count"] = [r[2] for r in results]

    # Map intent to label_id
    df["label"] = df["intent"].map(config.LABEL2ID)

    # Filter out near-empty text (< 3 characters or < 2 words)
    df = df[df["clean_text"].str.len() >= 3]
    df = df[df["clean_text"].apply(lambda x: len(x.split()) >= 1)]

    # Deduplicate on clean_text
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["clean_text", "intent"]).reset_index(drop=True)
    logger.info("Deduplication: dropped %d duplicate records. Retained: %d", before_dedup - len(df), len(df))

    return df


def split_dataset(
    df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train/val/test split based on intent label.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Step 1: Split into train and temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df, test_size=temp_ratio, random_state=seed, stratify=df["intent"]
    )

    # Step 2: Split temp into val and test
    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_ratio, random_state=seed, stratify=temp_df["intent"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def print_class_balance_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Logs comprehensive dataset summary and class distribution report.
    """
    print("\n" + "=" * 65)
    print("           DATASET CLASS DISTRIBUTION REPORT")
    print("=" * 65)
    print(f"Total Dataset Size: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"  • Train Set: {len(train_df)} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"  • Val Set:   {len(val_df)} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"  • Test Set:  {len(test_df)} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print("-" * 65)

    summary_data = []
    for label in config.INTENT_LABELS:
        tr_cnt = (train_df["intent"] == label).sum()
        vl_cnt = (val_df["intent"] == label).sum()
        ts_cnt = (test_df["intent"] == label).sum()
        total = tr_cnt + vl_cnt + ts_cnt
        summary_data.append({
            "Intent Label": label,
            "Train": tr_cnt,
            "Val": vl_cnt,
            "Test": ts_cnt,
            "Total": total,
            "Train %": f"{tr_cnt/len(train_df)*100:.1f}%",
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("=" * 65 + "\n")


def main():
    raw_path = config.RAW_DATA_DIR / "raw_dataset.csv"
    if not raw_path.exists():
        logger.info("Raw dataset not found at %s. Running load_dataset first...", raw_path)
        from src.data.load_dataset import main as run_load
        run_load()

    logger.info("Loading raw dataset from %s", raw_path)
    raw_df = pd.read_csv(raw_path)

    # Preprocess
    cleaned_df = preprocess_dataframe(raw_df)

    # Split 70 / 15 / 15
    train_df, val_df, test_df = split_dataset(
        cleaned_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=config.SEED
    )

    # Save to data/processed/
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False, encoding="utf-8")
    val_df.to_csv(config.VAL_DATA_PATH, index=False, encoding="utf-8")
    test_df.to_csv(config.TEST_DATA_PATH, index=False, encoding="utf-8")

    logger.info("Saved train.csv -> %s", config.TRAIN_DATA_PATH)
    logger.info("Saved val.csv   -> %s", config.VAL_DATA_PATH)
    logger.info("Saved test.csv  -> %s", config.TEST_DATA_PATH)

    # Print distribution
    print_class_balance_report(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
