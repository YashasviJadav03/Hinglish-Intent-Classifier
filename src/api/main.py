"""
src/api/main.py

FastAPI inference service for Hinglish Voice-Agent Intent Classification.

Features:
1. Low-memory container optimization (<512MB RAM) with PEFT LoRA adapter
2. Confidence Thresholding & Uncertainty Fallback Detection (flags predictions with max softmax < threshold)
3. Secondary Intent prediction for ambiguous conversational boundaries
4. Single-utterance (/classify) and Batch-vectorized (/classify/batch) inference endpoints
5. Interactive dashboard mounting with CORS support
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
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
            logger.warning("Could not load adapter from Hub: %s. Falling back to base model.", e)
            MODEL = base_model
            TOKENIZER = AutoTokenizer.from_pretrained(base_model_name)

    MODEL.to(DEVICE)
    MODEL.eval()
    gc.collect()
    logger.info("Model ready for inference!")
    return MODEL, TOKENIZER, DEVICE


class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The customer voice transcript utterance in code-mixed Hinglish.",
        example="Thoda discount de do na, price bohot zyada hai.",
    )
    confidence_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Threshold below which prediction is marked as uncertain / fallback.",
        example=0.60,
    )


class ClassifyBatchRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of Hinglish utterances to classify in batch.",
        example=["Thoda discount de do na", "Refund kab aayega?"],
    )
    confidence_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Threshold below which predictions are marked as uncertain / fallback.",
        example=0.60,
    )


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    is_uncertain: bool
    fallback: bool
    secondary_intent: Optional[str] = None
    secondary_confidence: Optional[float] = None
    cleaned_text: str
    all_scores: Dict[str, float]


class ClassifyBatchResponse(BaseModel):
    results: List[ClassifyResponse]
    total: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model into memory and perform warmup inference
    model, tokenizer, device = get_model_and_tokenizer()
    try:
        warmup_inputs = tokenizer("warmup query", return_tensors="pt", max_length=32, truncation=True).to(device)
        with torch.no_grad():
            _ = model(**warmup_inputs)
        logger.info("Model warmup inference completed successfully.")
    except Exception as e:
        logger.warning("Warmup failed (non-fatal): %s", e)
    yield


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="Production FastAPI service serving fine-tuned LoRA Transformer for code-mixed Hindi-English voice utterances with confidence fallback safeguards.",
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

# Mount static web app files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", tags=["UI"])
def root():
    """Serves the interactive web application dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/api/info", tags=["System"])
def api_info():
    """Returns API and service metadata in JSON."""
    return {
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "status": "online",
        "confidence_threshold_default": 0.60,
        "supported_intents": config.INTENT_LABELS,
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

    try:
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

        sorted_indices = probs.argsort()[::-1]
        top_idx = int(sorted_indices[0])
        predicted_intent = config.ID2LABEL[top_idx]
        confidence = float(round(float(probs[top_idx]), 4))

        # Secondary intent calculation
        secondary_idx = int(sorted_indices[1])
        secondary_intent = config.ID2LABEL[secondary_idx]
        secondary_confidence = float(round(float(probs[secondary_idx]), 4))

        # Confidence Thresholding & Uncertainty detection
        is_uncertain = confidence < request.confidence_threshold
        fallback = is_uncertain

        all_scores = {config.ID2LABEL[i]: float(round(float(prob), 4)) for i, prob in enumerate(probs)}

        return ClassifyResponse(
            intent=predicted_intent,
            confidence=confidence,
            is_uncertain=is_uncertain,
            fallback=fallback,
            secondary_intent=secondary_intent,
            secondary_confidence=secondary_confidence,
            cleaned_text=clean_text,
            all_scores=all_scores,
        )
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/classify/batch", response_model=ClassifyBatchResponse, tags=["Inference"])
def classify_batch(request: ClassifyBatchRequest):
    """Processes a batch of utterances in a single vectorized forward pass with fallback thresholding."""
    model, tokenizer, device = get_model_and_tokenizer()

    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")

    try:
        cleaned_texts = []
        for t in request.texts:
            clean_t, _, _ = extract_metadata_and_clean(t.strip())
            cleaned_texts.append(clean_t if clean_t else t.strip())

        inputs = tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs_matrix = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        results = []
        for i, probs in enumerate(probs_matrix):
            sorted_indices = probs.argsort()[::-1]
            top_idx = int(sorted_indices[0])
            predicted_intent = config.ID2LABEL[top_idx]
            confidence = float(round(float(probs[top_idx]), 4))

            secondary_idx = int(sorted_indices[1])
            secondary_intent = config.ID2LABEL[secondary_idx]
            secondary_confidence = float(round(float(probs[secondary_idx]), 4))

            is_uncertain = confidence < request.confidence_threshold
            fallback = is_uncertain

            all_scores = {config.ID2LABEL[j]: float(round(float(p), 4)) for j, p in enumerate(probs)}
            results.append(
                ClassifyResponse(
                    intent=predicted_intent,
                    confidence=confidence,
                    is_uncertain=is_uncertain,
                    fallback=fallback,
                    secondary_intent=secondary_intent,
                    secondary_confidence=secondary_confidence,
                    cleaned_text=cleaned_texts[i],
                    all_scores=all_scores,
                )
            )

        return ClassifyBatchResponse(results=results, total=len(results))
    except Exception as e:
        logger.error("Batch inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
