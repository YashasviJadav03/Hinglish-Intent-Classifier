"""
tests/test_preprocess.py

Unit tests for Hinglish data preprocessing, regex normalization,
emoji extraction, and stratified splitting logic.
"""

import pandas as pd
import pytest
from src.data.preprocess import extract_metadata_and_clean, preprocess_dataframe, split_dataset
import config


def test_extract_metadata_and_clean_elongations():
    """Verify transliteration elongation is compressed properly."""
    raw = "plzzzz help meeeeee boooohot discount chahiye"
    clean, emojis, excess_punct = extract_metadata_and_clean(raw)
    assert "plzz" in clean
    assert "meeeeee" not in clean
    assert "meee" not in clean
    assert "booo" not in clean


def test_extract_metadata_and_clean_emojis():
    """Verify emoji extraction and removal from cleaned text."""
    raw = "Ye product bahut accha hai 👍🙏🔥"
    clean, emojis, excess_punct = extract_metadata_and_clean(raw)
    assert "👍" in emojis
    assert "🙏" in emojis
    assert "👍" not in clean
    assert clean == "Ye product bahut accha hai"


def test_extract_metadata_and_clean_excess_punct():
    """Verify excess punctuation count and compression."""
    raw = "Order kyu nahi aaya?????"
    clean, emojis, excess_punct = extract_metadata_and_clean(raw)
    assert excess_punct >= 1
    assert "????" not in clean
    assert clean.endswith("?")


def test_extract_metadata_and_clean_empty_and_invalid():
    """Verify edge cases with non-strings and empty input."""
    assert extract_metadata_and_clean("") == ("", "", 0)
    assert extract_metadata_and_clean(None) == ("", "", 0)
    assert extract_metadata_and_clean("   ") == ("", "", 0)


def test_preprocess_dataframe():
    """Verify dataframe deduplication, filtering, and label mapping."""
    sample_data = {
        "text": [
            "Thoda discount de do na",
            "Thoda discount de do na",  # duplicate
            "ok",                       # too short (<3 chars)
            "Mera order deliver nahi hua",
            "Random text with invalid intent"
        ],
        "intent": [
            "price_negotiation",
            "price_negotiation",
            "positive_confirmation",
            "complaint",
            "unknown_intent_xyz"
        ]
    }
    df = pd.DataFrame(sample_data)
    cleaned_df = preprocess_dataframe(df)

    assert len(cleaned_df) == 2
    assert "clean_text" in cleaned_df.columns
    assert "label" in cleaned_df.columns
    assert set(cleaned_df["intent"]) == {"price_negotiation", "complaint"}


def test_split_dataset_stratification():
    """Verify stratified splitting proportions and labels."""
    intents = config.INTENT_LABELS * 20  # 120 samples
    df = pd.DataFrame({
        "clean_text": [f"Sample utterance number {i}" for i in range(len(intents))],
        "intent": intents,
        "label": [config.LABEL2ID[i] for i in intents]
    })
    train_df, val_df, test_df = split_dataset(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert len(train_df) == int(len(df) * 0.70)
    for intent in config.INTENT_LABELS:
        assert (train_df["intent"] == intent).sum() > 0
        assert (val_df["intent"] == intent).sum() > 0
        assert (test_df["intent"] == intent).sum() > 0
