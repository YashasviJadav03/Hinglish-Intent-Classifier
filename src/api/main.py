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
from contextlib import asynccontextmanager

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

# Global model state
MODEL = None
TOKENIZER = None
DEVICE = None


def get_model_and_tokenizer():
    """
    Loads base model and attaches the fine-tuned LoRA adapter if not already in memory.
    Optimized for low-memory container environments (<512MB RAM).
    """
    global MODEL, TOKENIZER, DEVICE
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER, DEVICE

    import gc
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    logger.info("Loading model and tokenizer with memory optimization...")
    DEVICE = torch.device("cpu")
    base_model_name = config.DEFAULT_MODEL_NAME
    adapter_path = config.LORA_ADAPTER_DIR

    adapter_exists = (
        adapter_path.exists()
        and (adapter_path / "adapter_config.json").exists()
    )

    if adapter_exists:
        logger.info("Loading fine-tuned LoRA adapter from local path %s", adapter_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=config.NUM_LABELS,
            id2label=config.ID2LABEL,
            label2id=config.LABEL2ID,
            low_cpu_mem_usage=True,
        )
        MODEL = PeftModel.from_pretrained(base_model, str(adapter_path))
        TOKENIZER = AutoTokenizer.from_pretrained(
            str(adapter_path) if (adapter_path / "tokenizer.json").exists() or (adapter_path / "vocab.txt").exists()
            else base_model_name
        )
    else:
        logger.info("Loading fine-tuned LoRA adapter directly from Hugging Face Hub: yashasvijadav03/hinglish-intent-classifier")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=config.NUM_LABELS,
            id2label=config.ID2LABEL,
            label2id=config.LABEL2ID,
            low_cpu_mem_usage=True,
        )
        try:
            MODEL = PeftModel.from_pretrained(base_model, "yashasvijadav03/hinglish-intent-classifier")
            TOKENIZER = AutoTokenizer.from_pretrained("yashasvijadav03/hinglish-intent-classifier")
        except Exception as e:
            logger.warning(f"Could not load adapter from Hub: {e}. Falling back to base model.")
            MODEL = base_model
            TOKENIZER = AutoTokenizer.from_pretrained(base_model_name)

    MODEL.to(DEVICE)
    MODEL.eval()
    gc.collect()
    logger.info("Model ready for inference!")
    return MODEL, TOKENIZER, DEVICE


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model into memory
    get_model_and_tokenizer()
    yield
    # Shutdown logic if needed


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="FastAPI service serving fine-tuned LoRA Transformer for noisy code-mixed Hindi-English voice utterances.",
    lifespan=lifespan,
)

# Enable CORS for web apps and dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    model, _, device = get_model_and_tokenizer()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "intent_classes": config.INTENT_LABELS,
    }


@app.post("/classify", response_model=ClassifyResponse, tags=["Inference"])
def classify_utterance(request: ClassifyRequest):
    model, tokenizer, device = get_model_and_tokenizer()

    raw_text = request.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Utterance text cannot be empty.")

    # Apply preprocessing normalizations
    clean_text, emojis, excess_punct = extract_metadata_and_clean(raw_text)
    if not clean_text:
        clean_text = raw_text

    # Tokenize
    inputs = tokenizer(
        clean_text,
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    predicted_intent = config.ID2LABEL[top_idx]
    confidence = float(round(float(probs[top_idx]), 4))

    all_scores = {config.ID2LABEL[i]: float(round(float(prob), 4)) for i, prob in enumerate(probs)}

    return ClassifyResponse(
        intent=predicted_intent,
        confidence=confidence,
        cleaned_text=clean_text,
        all_scores=all_scores,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
