"""
tests/test_api.py

Integration tests for FastAPI inference endpoints, batch classification,
confidence thresholding, fallback handling, secondary intent extraction,
and error validation using TestClient.
"""

from fastapi.testclient import TestClient
import pytest
from src.api.main import app
import config

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns 200 and valid schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "intent_classes" in data
    assert len(data["intent_classes"]) == 6


def test_api_info_endpoint():
    """Verify /api/info returns service metadata."""
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["status"] == "online"
    assert "confidence_threshold_default" in data


def test_classify_endpoint_valid():
    """Verify /classify returns prediction with probability distribution, fallback flags, and secondary intent."""
    payload = {"text": "Thoda discount de do na bhai price bohot zyada hai", "confidence_threshold": 0.60}
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "intent" in data
    assert data["intent"] in config.INTENT_LABELS
    assert 0.0 <= data["confidence"] <= 1.0
    assert "is_uncertain" in data
    assert "fallback" in data
    assert "secondary_intent" in data
    assert "secondary_confidence" in data
    assert "all_scores" in data
    assert len(data["all_scores"]) == 6
    assert abs(sum(data["all_scores"].values()) - 1.0) < 0.05


def test_classify_confidence_threshold_trigger():
    """Verify that a high threshold (e.g. 0.999) correctly triggers is_uncertain / fallback."""
    payload = {
        "text": "bhai phone kar lena kal",
        "confidence_threshold": 0.999  # very strict threshold
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    if data["confidence"] < 0.999:
        assert data["is_uncertain"] is True
        assert data["fallback"] is True


def test_classify_batch_endpoint():
    """Verify /classify/batch vector classification with thresholding."""
    payload = {
        "texts": [
            "Order deliver nahi hua, refund do",
            "Haanji done samjho payment link bhej do"
        ],
        "confidence_threshold": 0.50
    }
    response = client.post("/classify/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["intent"] in config.INTENT_LABELS
    assert data["results"][1]["intent"] in config.INTENT_LABELS
    assert "is_uncertain" in data["results"][0]
    assert "secondary_intent" in data["results"][0]


def test_classify_empty_string_error():
    """Verify 422/400 validation on empty payload."""
    response = client.post("/classify", json={"text": ""})
    assert response.status_code in [400, 422]


def test_root_serves_html():
    """Verify GET / returns HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
