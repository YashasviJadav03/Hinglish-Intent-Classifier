"""
src/model/compare_runs.py

Reads results/experiment_log.csv, sorts and formats all fine-tuning and ablation runs,
identifies the best performing LoRA configuration based on validation Macro-F1,
and outputs a clear summary table.
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compare_and_report_runs() -> pd.DataFrame:
    """
    Loads experiment log, prints sorted comparison table, and saves markdown summary.
    """
    log_path = config.EXPERIMENT_LOG_PATH
    if not log_path.exists():
        logger.error("No experiment log found at %s. Please run train.py first.", log_path)
        return None

    df = pd.read_csv(log_path)
    if df.empty:
        logger.warning("Experiment log is empty.")
        return df

    # Ensure required columns exist
    sort_col = "val_macro_f1" if "val_macro_f1" in df.columns else df.columns[-1]
    df_sorted = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    df_sorted["Rank"] = df_sorted.index + 1

    # Select columns for clean display
    display_cols = [
        "Rank",
        "run_name",
        "learning_rate",
        "lora_r",
        "lora_alpha",
        "epochs",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    ]
    present_cols = [c for c in display_cols if c in df_sorted.columns]
    table_df = df_sorted[present_cols]

    print("\n" + "=" * 80)
    print("                     LORA EXPERIMENTATION & ABLATION COMPARISON")
    print("=" * 80)
    print(table_df.to_string(index=False))
    print("-" * 80)

    best_run = df_sorted.iloc[0]
    print(f"★ BEST CONFIGURATION: {best_run.get('run_name')}")
    print(f"  • LoRA Rank (r) : {best_run.get('lora_r')}")
    print(f"  • Learning Rate : {best_run.get('learning_rate')}")
    print(f"  • Epochs        : {best_run.get('epochs')}")
    print(f"  • Val Macro-F1  : {best_run.get('val_macro_f1'):.4f}")
    print(f"  • Val Accuracy  : {best_run.get('val_accuracy'):.4f}")
    print(f"  • Adapter Path  : {best_run.get('adapter_dir', config.LORA_ADAPTER_DIR)}")
    print("=" * 80 + "\n")

    # Save summary to markdown
    md_summary_path = config.RESULTS_DIR / "ablation_summary.md"
    with open(md_summary_path, "w", encoding="utf-8") as f:
        f.write("# LoRA Fine-Tuning Ablation Study Summary\n\n")
        f.write(table_df.to_markdown(index=False))
        f.write(f"\n\n**Best Performing Run**: `{best_run.get('run_name')}` with **Macro-F1**: `{best_run.get('val_macro_f1')}`\n")
    logger.info("Saved ablation summary table to %s", md_summary_path)

    return df_sorted


def main():
    compare_and_report_runs()


if __name__ == "__main__":
    main()
