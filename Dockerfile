# Production-Ready Low-Memory Dockerfile for Hinglish Intent Classifier API
FROM python:3.10-slim

# Set environment variables for minimal memory footprint
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch first (saves ~2GB of CUDA bloat & prevents Exit 137 OOM)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining packages
COPY requirements.txt .
# Filter out torch from requirements.txt to keep CPU-only build
RUN pip install --no-cache-dir transformers peft fastapi "uvicorn[standard]" pydantic scikit-learn pandas numpy

# Copy source code and artifacts
COPY config.py .
COPY src/ src/
COPY data/ data/
COPY results/ results/

# Copy only the best LoRA adapter
COPY models/lora-adapter/adapter_config.json models/lora-adapter/
COPY models/lora-adapter/adapter_model.safetensors models/lora-adapter/
COPY models/lora-adapter/tokenizer.json models/lora-adapter/
COPY models/lora-adapter/tokenizer_config.json models/lora-adapter/

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose default port
EXPOSE 8000 7860 10000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run API server (dynamically binds to Render's $PORT, HF Spaces 7860, or local 8000)
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
