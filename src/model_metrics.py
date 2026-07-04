"""Evaluation metric helpers.

Small, dependency-light functions so the extra reporting (ordinal quality,
probability calibration, cross-validated stability) is unit-testable and shared
between ``train_model`` and ``evaluate``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, log_loss, mean_absolute_error


def ordinal_metrics(y_true, y_pred) -> dict:
    """Mean absolute error and quadratic-weighted kappa.

    Both reward predictions that land *near* the true rating, which is what we
    care about for an ordered 1-5 target -- plain accuracy treats "off by one"
    and "off by four" identically.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def expected_calibration_error(y_true, proba, classes, n_bins: int = 10) -> float:
    """Expected Calibration Error of the top-1 prediction.

    Bins predictions by confidence and averages |accuracy - confidence| weighted
    by bin population. 0 = perfectly calibrated.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    classes = np.asarray(classes)

    confidence = proba.max(axis=1)
    predictions = classes[proba.argmax(axis=1)]
    correct = (predictions == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence > lo) & (confidence <= hi)
        count = int(mask.sum())
        if count:
            ece += abs(correct[mask].mean() - confidence[mask].mean()) * count / n
    return float(ece)


def calibration_curve_points(y_true, proba, classes, n_bins: int = 10):
    """Return (mean_confidence, accuracy, count) per confidence bin for plotting."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    classes = np.asarray(classes)

    confidence = proba.max(axis=1)
    predictions = classes[proba.argmax(axis=1)]
    correct = (predictions == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf_pts, acc_pts, counts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum():
            conf_pts.append(float(confidence[mask].mean()))
            acc_pts.append(float(correct[mask].mean()))
            counts.append(int(mask.sum()))
    return np.array(conf_pts), np.array(acc_pts), np.array(counts)


def multiclass_log_loss(y_true, proba, classes) -> float:
    return float(log_loss(y_true, proba, labels=list(classes)))
