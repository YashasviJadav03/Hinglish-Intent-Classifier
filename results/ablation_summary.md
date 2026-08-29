# LoRA Fine-Tuning Ablation Study Summary

| Rank | run_name | learning_rate | lora_r | lora_alpha | epochs | val_loss | val_accuracy | val_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | lora_r16_lr5e4 | 0.0005 | 16 | 32 | 4 | 0.0254 | 1.0 | 1.0 |
| 2 | lora_r4_lr3e4 | 0.0003 | 4 | 8 | 5 | 0.2058 | 0.9444 | 0.9431 |
| 3 | lora_r8_lr3e4 | 0.0003 | 8 | 16 | 4 | 0.2705 | 0.9213 | 0.9207 |

**Best Performing Run**: `lora_r16_lr5e4` with **Macro-F1**: `1.0`
