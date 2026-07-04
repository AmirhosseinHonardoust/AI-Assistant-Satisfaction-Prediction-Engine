from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COL,
)
from .features import build_model, build_preprocessor
from .model_metrics import (
    calibration_curve_points,
    expected_calibration_error,
    ordinal_metrics,
)
from .ordinal import OrdinalClassifier


def load_model_and_data():
    model_path = MODELS_DIR / "satisfaction_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")

    model = joblib.load(model_path)
    test_path = PROCESSED_DATA_DIR / "sessions_test.csv"
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    return model, X_test, y_test


def plot_confusion_matrix(y_true, y_pred) -> None:
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cbar=True, ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Satisfaction Rating")

    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")


def plot_satisfaction_distribution(y_true, y_pred) -> None:
    fig, ax = plt.subplots()
    df_plot = pd.DataFrame({"true": y_true, "pred": y_pred})

    df_true = df_plot["true"].value_counts(normalize=True).sort_index()
    df_pred = df_plot["pred"].value_counts(normalize=True).sort_index()

    idx = sorted(set(df_true.index).union(df_pred.index))
    true_vals = [df_true.get(i, 0) for i in idx]
    pred_vals = [df_pred.get(i, 0) for i in idx]

    width = 0.35
    x = np.arange(len(idx))

    ax.bar(x - width / 2, true_vals, width, label="True")
    ax.bar(x + width / 2, pred_vals, width, label="Predicted")

    ax.set_xticks(x)
    ax.set_xticklabels(idx)
    ax.set_ylabel("Proportion")
    ax.set_title("Satisfaction Distribution: True vs Predicted")
    ax.legend()

    dist_path = FIGURES_DIR / "satisfaction_distribution.png"
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()
    print(f"Saved satisfaction distribution plot to: {dist_path}")


def plot_per_class_f1(report_dict) -> None:
    labels = [k for k in report_dict if k.isdigit()]
    f1_scores = [report_dict[k]["f1-score"] for k in labels]

    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    ax.bar(x, f1_scores)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1-score")
    ax.set_xlabel("Satisfaction Rating")
    ax.set_title("Per-Class F1-score")

    f1_path = FIGURES_DIR / "per_class_f1.png"
    plt.tight_layout()
    plt.savefig(f1_path)
    plt.close()
    print(f"Saved per-class F1 plot to: {f1_path}")


def plot_reliability_diagram(y_true, proba, classes) -> float:
    """Save a reliability diagram and return the Expected Calibration Error."""
    conf, acc, counts = calibration_curve_points(y_true, proba, classes)
    ece = expected_calibration_error(y_true, proba, classes)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    ax.plot(conf, acc, "o-", label="Model")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    path = FIGURES_DIR / "reliability_diagram.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved reliability diagram to: {path}  (ECE={ece:.3f})")
    return ece


def compare_ordinal_vs_flat() -> dict:
    """Fit a flat RF and a Frank-Hall ordinal RF on the same split and compare.

    Makes the "should this be ordinal regression?" question reproducible. On the
    synthetic data the flat forest wins on every metric, so it stays the default.
    """
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "sessions_train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "sessions_test.csv")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    pre = build_preprocessor().fit(X_train)
    Xtr, Xte = pre.transform(X_train), pre.transform(X_test)

    flat = build_model().fit(Xtr, y_train)
    flat_pred = flat.predict(Xte)

    ordinal = OrdinalClassifier(build_model()).fit(Xtr, y_train.values)
    ord_pred = ordinal.predict(Xte)

    def _summary(pred) -> dict:
        return {"accuracy": float(accuracy_score(y_test, pred)), **ordinal_metrics(y_test, pred)}

    result = {
        "flat_rf": _summary(flat_pred),
        "ordinal_frank_hall": _summary(ord_pred),
    }
    out = METRICS_DIR / "ordinal_comparison.json"
    with out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved ordinal-vs-flat comparison to: {out}")
    print(f"  flat RF   : {result['flat_rf']}")
    print(f"  ordinal   : {result['ordinal_frank_hall']}")
    return result


def main() -> None:
    model, X_test, y_test = load_model_and_data()
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    plot_confusion_matrix(y_test, y_pred)
    plot_satisfaction_distribution(y_test, y_pred)
    plot_reliability_diagram(y_test, proba, model.classes_)

    report = classification_report(y_test, y_pred, output_dict=True, digits=3)
    report_path = METRICS_DIR / "classification_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved updated classification report to: {report_path}")

    plot_per_class_f1(report)
    compare_ordinal_vs_flat()


if __name__ == "__main__":
    main()
