from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

from .config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, TARGET_COL
from .features import build_pipeline


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
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

    # Honest reference point: a majority-class baseline for this 5-class task.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

    cls_report = classification_report(y_test, y_pred, output_dict=True, digits=3)

    metrics = {
        "accuracy": float(acc),
        "baseline_accuracy": float(baseline_acc),
        "lift_over_baseline": float(acc - baseline_acc),
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
    print(f"Test accuracy:    {acc:.4f}  (baseline {baseline_acc:.4f})")

    return metrics


def main() -> None:
    metrics = train_and_evaluate()
    print("\n=== Metrics summary ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
