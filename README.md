# Hinglish Intent Classifier — Fine-Tuned Transformer for Code-Mixed Voice-Agent NLU

A lightweight, high-performance Intent Classification system for Hindi-English (Hinglish) conversational utterances designed for Voice-Agent NLU pipelines.

## Project Structure

```
hinglish-intent-classifier/
├── data/
│   ├── raw/                # Raw downloaded datasets
│   └── processed/          # Cleaned, normalized, and split CSVs (train/val/test)
├── models/
│   └── lora-adapter/       # Saved PEFT LoRA checkpoints and adapter weights
├── notebooks/              # Jupyter notebooks for EDA and error inspection
├── results/                # Evaluation reports, confusion matrices, ablation logs
├── src/
│   ├── api/                # FastAPI application and inference service
│   ├── data/               # Data loaders, cleaners, and split scripts
│   └── model/              # Zero-shot baseline, LoRA training, and evaluation scripts
├── .gitignore
├── config.py               # Global configurations, paths, and hyperparameters
├── requirements.txt        # Python package dependencies
└── README.md
```

## Quickstart & Environment Setup

### 1. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Project Phases
- **Phase 0**: Repo Scaffolding & Environment Setup
- **Phase 1**: Dataset Acquisition & Preprocessing
- **Phase 2**: Zero-Shot Baseline Evaluation
- **Phase 3**: LoRA Fine-Tuning Setup
- **Phase 4**: Experimentation & Ablation Study
- **Phase 5**: Final Evaluation & Qualitative Error Analysis
- **Phase 6**: FastAPI Deployment & Containerization
- **Phase 7**: Comprehensive Documentation & Resume Write-up
