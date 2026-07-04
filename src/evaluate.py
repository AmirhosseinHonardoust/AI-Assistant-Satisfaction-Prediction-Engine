from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COL,
)


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


def main() -> None:
    model, X_test, y_test = load_model_and_data()
    y_pred = model.predict(X_test)

    plot_confusion_matrix(y_test, y_pred)
    plot_satisfaction_distribution(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True, digits=3)
    report_path = METRICS_DIR / "classification_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved updated classification report to: {report_path}")

    plot_per_class_f1(report)


if __name__ == "__main__":
    main()
