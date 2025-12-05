import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from .config import PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR, TARGET_COL
from .features import build_pipeline


def load_processed():
    train_path = PROCESSED_DATA_DIR / "sessions_train.csv"
    test_path = PROCESSED_DATA_DIR / "sessions_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed files not found in {PROCESSED_DATA_DIR}. Run data_prep.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def train_and_evaluate() -> dict:
    train_df, test_df = load_processed()

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cls_report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        digits=3,
    )

    metrics = {
        "accuracy": float(acc),
        "n_test_samples": int(len(y_test)),
    }

    model_path = MODELS_DIR / "satisfaction_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    metrics_path = METRICS_DIR / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    report_path = METRICS_DIR / "classification_report.json"
    with report_path.open("w") as f:
        json.dump(cls_report, f, indent=2)

    print(f"Saved model to:   {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved report to:  {report_path}")
    print(f"Test accuracy:    {acc:.4f}")

    return metrics


def main() -> None:
    metrics = train_and_evaluate()
    print("\n=== Metrics summary ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
