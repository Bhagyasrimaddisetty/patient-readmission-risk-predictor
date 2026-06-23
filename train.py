"""Entry-point script: generate data → preprocess → train → save artefacts."""
import subprocess, sys
from pathlib import Path

import pandas as pd

# ── ensure package is importable when run from project root ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.preprocessing import load_data, preprocess_pipeline, TARGET_COL, get_feature_columns
from src.pipeline.model import (
    train_and_evaluate,
    select_best_model,
    save_artifact,
    save_results_json,
)


def main():
    raw_csv = Path("data/raw/patients.csv")

    # 1. Generate synthetic data if it doesn't exist
    if not raw_csv.exists():
        print("Generating synthetic data …")
        subprocess.run([sys.executable, "data/generate_data.py"], cwd=".", check=True)

    # 2. Load & preprocess
    print("\nLoading and preprocessing data …")
    df_raw = load_data(str(raw_csv))
    df, encoders, scaler = preprocess_pipeline(df_raw, fit=True)

    feature_cols = get_feature_columns(df)
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Class distribution:\n{df[TARGET_COL].value_counts()}")

    # 3. Train & evaluate
    results = train_and_evaluate(df)

    # 4. Select & persist best model
    best_name, best_model = select_best_model(results)
    save_artifact(best_model, "best_model.pkl")
    save_artifact(encoders, "encoders.pkl")
    save_artifact(scaler, "scaler.pkl")
    save_artifact(feature_cols, "feature_cols.pkl")
    save_results_json(results)

    print("\nTraining complete. Artefacts saved to models/")


if __name__ == "__main__":
    main()
