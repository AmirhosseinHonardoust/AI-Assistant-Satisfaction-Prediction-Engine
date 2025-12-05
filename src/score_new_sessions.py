from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from .config import MODELS_DIR, TIME_COL


def load_model():
    model_path = MODELS_DIR / "satisfaction_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first."
        )
    model = joblib.load(model_path)
    return model


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if TIME_COL not in df.columns:
        return df

    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df["hour_of_day"] = df[TIME_COL].dt.hour
    df["day_of_week"] = df[TIME_COL].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def score_file(input_csv: str | Path, output_csv: Optional[str | Path] = None) -> Path:
    model = load_model()

    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)
    df = engineer_time_features(df)

    # Predict classes and probabilities
    preds = model.predict(df)
    proba = model.predict_proba(df)

    df_scored = df.copy()
    df_scored["pred_satisfaction"] = preds

    classes = model.classes_
    for i, c in enumerate(classes):
        df_scored[f"p_rating_{c}"] = proba[:, i]

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_scored.csv")

    output_csv = Path(output_csv)
    df_scored.to_csv(output_csv, index=False)
    print(f"Saved scored sessions to: {output_csv}")
    return output_csv


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Score new AI assistant sessions for satisfaction.")
    parser.add_argument("input_csv", type=str, help="Path to CSV file with sessions.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional output CSV path; defaults to <input>_scored.csv",
    )
    args = parser.parse_args()

    score_file(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
