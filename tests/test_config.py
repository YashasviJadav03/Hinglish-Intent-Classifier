"""
tests/test_config.py

Unit tests for configuration consistency, label mappings, and directory paths.
"""

import config


def test_intent_labels_and_mappings():
    """Verify bidirectional consistency between ID2LABEL and LABEL2ID."""
    assert len(config.INTENT_LABELS) == config.NUM_LABELS
    assert len(config.ID2LABEL) == config.NUM_LABELS
    assert len(config.LABEL2ID) == config.NUM_LABELS

    for idx, label in config.ID2LABEL.items():
        assert config.LABEL2ID[label] == idx

    expected_labels = {
        "complaint",
        "purchase_inquiry",
        "price_negotiation",
        "callback_request",
        "not_interested",
        "positive_confirmation",
    }
    assert set(config.INTENT_LABELS) == expected_labels


def test_hyperparameters():
    """Verify essential training hyperparameter constants."""
    assert config.MAX_LENGTH == 128
    assert config.SEED == 42
    assert "r" in config.DEFAULT_LORA_PARAMS
    assert "lora_alpha" in config.DEFAULT_LORA_PARAMS
