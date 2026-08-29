"""
src/api/main.py

FastAPI inference service for Hinglish Voice-Agent Intent Classification.

Endpoints:
- GET  /health   -> Health check & model metadata
- POST /classify -> Classifies Hinglish utterance and returns top intent, confidence, and full probability distribution
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import config
from src.data.preprocess import extract_metadata_and_clean

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hinglish_api")

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="FastAPI service serving fine-tuned LoRA Transformer for noisy code-mixed Hindi-English voice utterances.",
)

# Enable CORS for web apps and dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL = None
TOKENIZER = None
DEVICE = None


class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="The customer voice transcript utterance in code-mixed Hinglish.",
        example="Thoda discount de do na, price bohot zyada hai.",
    )


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    cleaned_text: str
    all_scores: Dict[str, float]


@app.on_event("startup")
def load_model():
    """
    Initializes tokenizer and model on server startup.
    """
    global MODEL, TOKENIZER, DEVICE
    logger.info("Initializing Hinglish Intent Classifier service...")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Serving on compute device: %s", DEVICE)

    base_model_name = config.DEFAULT_MODEL_NAME
    adapter_path = config.LORA_ADAPTER_DIR

    try:
        if adapter_path.exists() and any(adapter_path.iterdir()):
            logger.info("Loading fine-tuned LoRA adapter from %s", adapter_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=config.NUM_LABELS,
                id2label=config.ID2LABEL,
                label2id=config.LABEL2ID,
            )
            MODEL = PeftModel.from_pretrained(base_model, str(adapter_path))
            TOKENIZER = AutoTokenizer.from_pretrained(str(adapter_path) if (adapter_path / "vocab.txt").exists() else base_model_name)
        else:
            logger.warning("LoRA adapter not found. Loading base checkpoint: %s", base_model_name)
            MODEL = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=config.NUM_LABELS,
                id2label=config.ID2LABEL,
                label2id=config.LABEL2ID,
            )
            TOKENIZER = AutoTokenizer.from_pretrained(base_model_name)

        MODEL.to(DEVICE)
        MODEL.eval()
        logger.info("Model loaded and ready for inference!")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise RuntimeError(f"Model initialization failed: {e}")


@app.get("/", tags=["System"])
def root():
    return {
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "intent_classes": config.INTENT_LABELS,
    }


@app.post("/classify", response_model=ClassifyResponse, tags=["Inference"])
def classify_utterance(request: ClassifyRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or initializing.")

    raw_text = request.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Utterance text cannot be empty.")

    # Apply preprocessing normalizations
    clean_text, emojis, excess_punct = extract_metadata_and_clean(raw_text)
    if not clean_text:
        clean_text = raw_text

    # Tokenize
    inputs = TOKENIZER(
        clean_text,
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    predicted_intent = config.ID2LABEL[top_idx]
    confidence = float(round(probs[top_idx], 4))

    all_scores = {config.ID2LABEL[i]: float(round(prob, 4)) for i, prob in enumerate(probs)}

    return ClassifyResponse(
        intent=predicted_intent,
        confidence=confidence,
        cleaned_text=clean_text,
        all_scores=all_scores,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
