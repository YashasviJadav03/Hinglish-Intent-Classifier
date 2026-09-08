"""
tests/test_preprocess.py

Unit tests for Hinglish data preprocessing, regex normalization,
emoji extraction, group-based stratified splitting, augmentation isolation,
and zero-leakage assertions.
"""

import pandas as pd
import pytest
from src.data.preprocess import (
    extract_metadata_and_clean,
    preprocess_dataframe,
    split_dataset_by_base_id,
    augment_training_data,
    verify_split_leakage,
)
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
    assert "base_id" in cleaned_df.columns
    assert set(cleaned_df["intent"]) == {"price_negotiation", "complaint"}


def test_split_dataset_by_base_id_zero_leakage():
    """Verify that split_dataset_by_base_id strictly partitions base_ids with zero overlap."""
    intents = config.INTENT_LABELS * 20  # 120 samples
    df = pd.DataFrame({
        "base_id": [f"id_{i:04d}" for i in range(len(intents))],
        "text": [f"Sample utterance base {i}" for i in range(len(intents))],
        "clean_text": [f"Sample utterance base {i}" for i in range(len(intents))],
        "intent": intents,
        "label": [config.LABEL2ID[i] for i in intents],
        "emojis": ["" for _ in range(len(intents))],
        "excess_punct_count": [0 for _ in range(len(intents))],
    })
    train_df, val_df, test_df = split_dataset_by_base_id(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    # 1. Total samples check
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # 2. Strict ID disjointness check (Zero Leakage)
    train_ids = set(train_df["base_id"])
    val_ids = set(val_df["base_id"])
    test_ids = set(test_df["base_id"])

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # 3. Stratification balance
    for intent in config.INTENT_LABELS:
        assert (train_df["intent"] == intent).sum() > 0
        assert (val_df["intent"] == intent).sum() > 0
        assert (test_df["intent"] == intent).sum() > 0


def test_augment_training_data_isolation():
    """Verify that augmentation expands only the provided training dataset."""
    sample_df = pd.DataFrame({
        "base_id": ["comp_001", "price_001"],
        "text": ["Mera refund kyu nahi aaya", "Thoda discount de do na"],
        "intent": ["complaint", "price_negotiation"],
        "clean_text": ["Mera refund kyu nahi aaya", "Thoda discount de do na"],
        "emojis": ["", ""],
        "excess_punct_count": [0, 0],
        "label": [0, 2],
    })

    aug_df = augment_training_data(sample_df)
    # Default variations has 8 combinations -> 2 * 8 = 16 rows
    assert len(aug_df) == 16
    assert set(aug_df["base_id"]) == {"comp_001", "price_001"}
    assert "clean_text" in aug_df.columns
    assert "label" in aug_df.columns


def test_verify_split_leakage_assertion():
    """Verify that verify_split_leakage accepts disjoint splits and raises errors on leakage."""
    valid_train = pd.DataFrame({
        "base_id": ["base_1", "base_2"],
        "clean_text": ["train phrase one", "train phrase two"],
    })
    valid_val = pd.DataFrame({
        "base_id": ["base_3"],
        "clean_text": ["val phrase three"],
    })
    valid_test = pd.DataFrame({
        "base_id": ["base_4"],
        "clean_text": ["test phrase four"],
    })

    # Should pass without exception
    assert verify_split_leakage(valid_train, valid_val, valid_test) is True

    # Artificially inject ID leakage
    leaky_test = pd.DataFrame({
        "base_id": ["base_1"],  # Overlaps with train!
        "clean_text": ["different text but same id"],
    })
    with pytest.raises(ValueError, match="CRITICAL LEAKAGE"):
        verify_split_leakage(valid_train, valid_val, leaky_test)

    # Artificially inject exact clean_text leakage
    leaky_text_test = pd.DataFrame({
        "base_id": ["base_99"],
        "clean_text": ["train phrase one"],  # Overlaps with train text!
    })
    with pytest.raises(ValueError, match="CRITICAL LEAKAGE"):
        verify_split_leakage(valid_train, valid_val, leaky_text_test)
