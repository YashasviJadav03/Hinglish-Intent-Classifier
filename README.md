# Hinglish Voice-Agent Intent Classifier

<div align="center">

[![CI Pipeline](https://github.com/YashasviJadav03/Hinglish-Intent-Classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/YashasviJadav03/Hinglish-Intent-Classifier/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace PEFT](https://img.shields.io/badge/🤗%20PEFT-LoRA-yellow.svg)](https://huggingface.co/docs/peft)
[![Docker Ready](https://img.shields.io/badge/docker-ready-2496ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)

**Production-grade, parameter-efficient NLP pipeline for Romanized Hindi-English (Hinglish) voice-agent transcripts, fine-tuned with PEFT LoRA, zero-leakage group splitting, OOD benchmarking, and confidence fallback safeguards.**

[🌐 Live Web Demo](https://hinglish-intent-classifier.onrender.com/) • [📑 API Docs (Swagger)](https://hinglish-intent-classifier.onrender.com/docs) • [🤗 Hugging Face Model](https://huggingface.co/yashasvijadav03/hinglish-intent-classifier)

</div>

---

## 1. Problem & Engineering Highlights

Conversational voice agents in South Asia process code-mixed speech where users interleave Hindi grammar and English vocabulary (*e.g., "Thoda discount de do na, price bohot zyada lag raha hai"* or *"Abhi drive kar raha hu, 6 PM call back karna"*). Monolingual models fail due to phonetic spelling variations (`chahiye` / `chaiye`), conversational boundary overlap, and ASR noise.

### Key Engineering Decisions:
- **Zero-Leakage Group Splitting**: Eliminated pre-split augmentation leakage by partitioning base utterances by `base_id` *before* applying synthetic variations.
- **Multi-Model Benchmark**: Evaluated classical ML (TF-IDF + SVM/LR/NB), zero-shot heuristics, and fine-tuned DistilBERT + PEFT LoRA on both In-Domain and noisy Out-of-Distribution (OOD) test sets.
- **Indic Tokenizer Fertility Analysis**: Quantified subword fragmentation across multilingual backbones (`DistilBERT` vs `Google MuRIL` vs `L3Cube Hing-BERT`).
- **Production Fallback Engine**: FastAPI microservice with configurable confidence thresholding (`threshold < 0.60`) to route ambiguous queries safely.

---

## 2. Validation Architecture & Zero-Leakage Pipeline

```mermaid
flowchart TD
    A["Curated Base Utterances (840 seeds)"] --> B["Group-Based Stratified Split (by base_id)"]
    B -->|70% Train Seeds (588)| C["Isolated Train Augmentation (4,704 samples)"]
    B -->|15% Val Seeds (126)| D["Clean Val Set (126 un-augmented)"]
    B -->|15% Test Seeds (126)| E["Clean In-Domain Test Set (126 un-augmented)"]
    F["Dedicated Independent OOD Corpus"] --> G["Clean OOD Benchmark (120 noisy ASR / slang)"]
    C --> H["DistilBERT + PEFT LoRA"]
    H --> E
    H --> G
```

- **Leakage Prevention**: Synthetic variations are strictly isolated to `train.csv`. Validation and test sets contain zero augmented artifacts.
- **CI Quality Gates**: Automated CI checks assert $\text{Train}_{\text{base\_id}} \cap \text{Test}_{\text{base\_id}} = \emptyset$ and enforce realistic macro F1 bounds ($0.75 \le \text{F1} \le 0.99$).

---

## 3. Benchmark Results & Model Comparison

Models benchmarked on **In-Domain Clean Test Set** ($N=126$) and **Noisy OOD Benchmark** ($N=120$):

| Architecture | In-Domain Acc | In-Domain Macro F1 | OOD Acc | OOD Macro F1 | CPU Latency | Memory Footprint | Role in Production |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TF-IDF + Logistic Regression** | **93.7%** | **0.9370** | **92.5%** | **0.9237** | **2 ms** | < 5 MB | Ultra-low latency primary / edge filter |
| **TF-IDF + Linear SVM** | 92.9% | 0.9287 | 93.3% | 0.9320 | 2 ms | < 5 MB | Strong linear baseline |
| **TF-IDF + Naive Bayes** | 91.3% | 0.9124 | 90.0% | 0.8943 | 2 ms | < 5 MB | Fast probabilistic baseline |
| **TF-IDF + Random Forest** | 92.1% | 0.9210 | 75.0% | 0.7575 | 12 ms | 35 MB | Overfits on unseen OOD phrasing |
| **Zero-Shot Keyword Heuristic** | 61.9% | 0.5973 | 60.0% | 0.5649 | 1 ms | < 1 MB | Naive rule-based lower bound |
| **Fine-Tuned DistilBERT + LoRA** | **88.9%** | **0.8886** | **78.3%** | **0.7831** | **35 ms** | **270 MB** | Context-aware deep representation |

<div align="center">
  <img src="results/confusion_matrix.png" alt="Normalized Confusion Matrix" width="600"/>
</div>

---

## 4. Indic Backbone & Tokenizer Fertility Study

Subword tokenization fragmentation on Hinglish roots across multilingual backbones (from [results/ablation_summary.md](results/ablation_summary.md)):

| Backbone | Parameters | Vocab Size | Subwords / Hinglish Word | Avg Tokens / Utterance | CPU Latency | Architectural Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`distilbert-base-multilingual-cased`** | **135M** | **119k** | **2.90** | **18.4** | **35 ms** | **Optimal latency-accuracy trade-off for voice streaming** |
| **`google/muril-base-cased`** | 236M | 197k | 2.20 | 14.6 | 68 ms | Lowest fertility (native Indic vocab); higher CPU latency |
| **`l3cube-pune/hing-bert`** | 110M | 120k | 3.30 | 18.2 | 30 ms | Fast Hinglish-specific model; higher subword fragmentation |
| **`l3cube-pune/hing-roberta`** | 125M | 50k | 2.45 | 15.6 | 32 ms | Compact BPE representation |

---

## 5. Error Analysis & Linguistic Failure Modes

Audited from [results/misclassified_examples.csv](results/misclassified_examples.csv):

1. **Negative Sentiment Spillover**:
   - *Sample*: `"bhai itna ghatiya product life me ni dekha return lelo isko"` (True: `complaint` $\to$ Pred: `not_interested`).
   - *Root Cause*: High negative sentiment tokens bleed into the refusal class rather than the complaint pipeline.
2. **Implicit Inquiries vs Imperative Bargaining**:
   - *Sample*: `"First time customer ke liye koi introductory discount voucher hai?"` (True: `price_negotiation` $\to$ Pred: `purchase_inquiry`).
   - *Root Cause*: Grammatically structured as an information inquiry rather than an explicit bargaining command.
3. **Multi-Intent Code-Switching**:
   - *Sample*: `"Daily 10 call aate hain aapke, block list me daal raha hu number"` (True: `not_interested` $\to$ Pred: `callback_request`).
   - *Root Cause*: Density of calling-related tokens (`"call"`, `"number"`) without explicit refusal trigger words.

---

## 6. Production Microservice & Uncertainty Fallback

The FastAPI service in [src/api/main.py](src/api/main.py) implements automated uncertainty detection:

```bash
# Predict intent with confidence thresholding (0.60 default)
curl -X POST "http://localhost:7860/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "Is model me discount mil sakta hai kya?", "confidence_threshold": 0.60}'
```

```json
{
  "intent": "price_negotiation",
  "confidence": 0.5421,
  "is_uncertain": true,
  "fallback": true,
  "secondary_intent": "purchase_inquiry",
  "secondary_confidence": 0.4103,
  "cleaned_text": "Is model me discount mil sakta hai kya?"
}
```

---

## 7. Interactive Web Dashboard

The [live web demo](https://hinglish-intent-classifier.onrender.com/) includes:

- **Single Utterance Classifier** — Type or select sample Hinglish utterances, adjust the confidence fallback threshold via slider, and see real-time intent prediction with softmax probability distribution bars and a circular confidence gauge.
- **Animated Project Stats** — Key metrics (840 base utterances, 5,076 total samples, 6 intent classes, 88.9% F1, 78.3% OOD F1, 35ms latency) animate into view on scroll.
- **Batch Demo** — Fires 6 diverse utterances (one per intent class) simultaneously via `/classify/batch` and renders results in a staggered animated card grid with fallback detection.
- **Uncertainty Fallback Safeguards** — When model confidence drops below the configurable threshold, an amber warning banner surfaces the secondary intent recommendation.

---

## 8. Quickstart

```bash
# 1. Install dependencies
git clone https://github.com/YashasviJadav03/Hinglish-Intent-Classifier.git
cd Hinglish-Intent-Classifier && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run zero-leakage pipeline & evaluations
python src/data/load_dataset.py && python src/data/preprocess.py
python src/model/baseline_eval.py
python src/model/evaluate.py
pytest tests/ -v

# 3. Launch FastAPI Server & UI
uvicorn src.api.main:app --host 0.0.0.0 --port 7860 --reload
```

### Docker
```bash
docker build -t hinglish-intent-classifier .
docker run -p 7860:7860 hinglish-intent-classifier
```
