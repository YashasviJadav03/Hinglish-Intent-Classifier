# Baseline vs Fine-Tuned Model Performance Comparison

| Metric | Zero-Shot Baseline (DistilBERT NLI) | Fine-Tuned (DistilBERT + PEFT LoRA) | Absolute Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **39.35%** | **100.00%** | **+60.65% pts** |
| **Macro F1-Score** | **0.3391** | **1.0000** | **+66.09% pts** |
| **Weighted F1-Score** | 0.3391 | 1.0000 | +66.09% pts |

## Per-Class Breakdown

| Intent Class | Baseline F1 | LoRA Fine-Tuned F1 | Delta F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| `complaint` | 0.2917 | **1.0000** | +0.7083 | 36 |
| `purchase_inquiry` | 0.5106 | **1.0000** | +0.4894 | 36 |
| `price_negotiation` | 0.3505 | **1.0000** | +0.6495 | 36 |
| `callback_request` | 0.0541 | **1.0000** | +0.9459 | 36 |
| `not_interested` | 0.2857 | **1.0000** | +0.7143 | 36 |
| `positive_confirmation` | 0.5421 | **1.0000** | +0.4579 | 36 |

> **Key Takeaway**: LoRA fine-tuning significantly resolves dialectal ambiguities and noisy code-mixed phonetics that off-the-shelf zero-shot NLI models fail to disambiguate.
