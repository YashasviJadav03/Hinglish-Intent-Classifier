# Hugging Face Spaces Dockerfile for Hinglish Intent Classifier API
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 appuser

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY config.py .
COPY src/ src/
COPY data/ data/
COPY results/ results/

# Copy only the best LoRA adapter (not all ablation checkpoints)
COPY models/lora-adapter/adapter_config.json models/lora-adapter/
COPY models/lora-adapter/adapter_model.safetensors models/lora-adapter/
COPY models/lora-adapter/tokenizer.json models/lora-adapter/
COPY models/lora-adapter/tokenizer_config.json models/lora-adapter/

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# HF Spaces requires port 7860
EXPOSE 7860

# Run API server on port 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
