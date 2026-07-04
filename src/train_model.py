from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

from .config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COL
from .features import build_pipeline
from .model_metrics import multiclass_log_loss, ordinal_metrics


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
    proba = pipeline.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Honest reference point: a majority-class baseline for this 5-class task.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

    # Cross-validated accuracy on all data: shows the single-split number is not
    # a lucky fold. Uses a fresh pipeline per fold.
    all_X = pd.concat([X_train, X_test])
    all_y = pd.concat([y_train, y_test])
    cv_scores = cross_val_score(build_pipeline(), all_X, all_y, cv=5)

    ord_m = ordinal_metrics(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, output_dict=True, digits=3)

    metrics = {
        "accuracy": float(acc),
        "baseline_accuracy": float(baseline_acc),
        "lift_over_baseline": float(acc - baseline_acc),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "mae": ord_m["mae"],
        "quadratic_weighted_kappa": ord_m["quadratic_weighted_kappa"],
        "log_loss": multiclass_log_loss(y_test, proba, pipeline.classes_),
        "n_test_samples": int(len(y_test)),
        "random_state": RANDOM_STATE,
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
    print(
        f"Test acc: {acc:.4f} (baseline {baseline_acc:.4f}) | "
        f"CV {cv_scores.mean():.3f}+/-{cv_scores.std():.3f} | "
        f"QWK {ord_m['quadratic_weighted_kappa']:.3f} | MAE {ord_m['mae']:.3f}"
    )

    return metrics


def main() -> None:
    metrics = train_and_evaluate()
    print("\n=== Metrics summary ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
