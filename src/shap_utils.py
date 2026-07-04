"""SHAP output-shape helpers.

SHAP changed its multiclass return convention across versions:

* legacy: a ``list`` of per-class ``(n_samples, n_features)`` arrays;
* modern (>= ~0.43): a single ``(n_samples, n_features, n_classes)`` ndarray.

These helpers normalise both so the explainability code is version-robust. They
depend only on numpy, which keeps them unit-testable without the heavy
``shap``/``streamlit`` stack installed.
"""

from __future__ import annotations

import numpy as np


def mean_abs_shap_across_classes(shap_values) -> np.ndarray:
    """Return mean(|SHAP|) over classes as a 2-D ``(n_samples, n_features)`` array."""
    if isinstance(shap_values, list):
        return np.mean([np.abs(sv) for sv in shap_values], axis=0)

    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        return np.abs(arr).mean(axis=2)
    return np.abs(arr)


def shap_row_for_class(shap_values, row: int, class_index: int) -> np.ndarray:
    """Extract a 1-D per-feature SHAP vector for one row and one class."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index][row]).reshape(-1)

    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_rows, n_features, n_classes)
        return arr[row, :, class_index].reshape(-1)
    return arr[row].reshape(-1)
