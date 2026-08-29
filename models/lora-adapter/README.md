---
license: apache-2.0
base_model: distilbert-base-multilingual-cased
library_name: peft
pipeline_tag: text-classification
language:
- hi
- en
tags:
- lora
- peft
- transformers
- intent-classification
- hinglish
- code-mixed
- nlu
- voice-agent
widget:
- text: "Thoda discount de do na, price bohot zyada hai."
  example_title: "Price Negotiation"
- text: "Order deliver nahi hua abhi tak, refund kab milega?"
  example_title: "Complaint"
- text: "Haanji done samjho, payment link share kar dijiye."
  example_title: "Positive Confirmation"
- text: "Abhi driving kar raha hoon, baad me phone karna."
  example_title: "Callback Request"
- text: "Mujhe ye product bilkul nahi chahiye, do not call."
  example_title: "Not Interested"
- text: "Bhaiya is plan ke features aur warranty explain kardo."
  example_title: "Purchase Inquiry"
---

# 🎙️ Hinglish Intent Classifier (LoRA + DistilBERT)

Fine-tuned **DistilBERT Multilingual** adapter using **LoRA (PEFT)** for **Intent Classification on Code-Mixed Hindi-English (Hinglish)** conversational utterances. Built for voice-agent Natural Language Understanding (NLU) pipelines.

## 🎯 Target Intent Classes
1. `complaint` (e.g., service issues, refund delays)
2. `purchase_inquiry` (e.g., catalog questions, warranty inquiries)
3. `price_negotiation` (e.g., asking for discounts, coupon requests)
4. `callback_request` (e.g., busy right now, call later)
5. `not_interested` (e.g., refusal, DND requests)
6. `positive_confirmation` (e.g., deal acceptance, confirmation)

## 📊 Benchmark Results

| Model / Configuration | Test Accuracy | Macro-F1 | Precision | Recall |
|---|---|---|---|---|
| **Zero-Shot Baseline (MNLI)** | 38.89% | 0.3540 | 0.4497 | 0.3889 |
| **LoRA Fine-Tuned (Rank 16, lr=5e-4)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** |

## 🚀 Quick Start & Usage

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

base_model_name = "distilbert-base-multilingual-cased"
adapter_repo = "yashasvijadav03/hinglish-intent-classifier"

# Intent labels mapping
id2label = {
    0: "complaint",
    1: "purchase_inquiry",
    2: "price_negotiation",
    3: "callback_request",
    4: "not_interested",
    5: "positive_confirmation"
}

# 1. Load Tokenizer & Base Model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=len(id2label),
    id2label=id2label,
)

# 2. Attach Fine-Tuned LoRA Adapter
model = PeftModel.from_pretrained(base_model, adapter_repo)
model.eval()

# 3. Classify an Utterance
text = "Thoda discount de do na, price bohot zyada hai."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)

predicted_id = torch.argmax(probs, dim=-1).item()
print(f"Predicted Intent: {id2label[predicted_id]} (Confidence: {probs[0][predicted_id]:.2%})")
# Output: Predicted Intent: price_negotiation (Confidence: 98.40%)
```

## 🛠️ Training Specifications
- **Base Model:** `distilbert-base-multilingual-cased` (135M params)
- **Trainable Parameters:** 1,185,798 (~0.87% of total model params via LoRA)
- **LoRA Parameters:** Rank $r=16$, Alpha $\alpha=32$, Dropout $0.1$, Target Modules: `q_lin`, `v_lin`
- **Training Epochs:** 3
- **Learning Rate:** `5e-4` with Linear Warmup