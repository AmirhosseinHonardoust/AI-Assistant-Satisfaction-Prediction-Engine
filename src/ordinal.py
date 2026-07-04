"""Frank & Hall ordinal classifier.

A light wrapper that turns a binary probabilistic classifier into an ordinal
one by training ``K-1`` models for ``P(y > k)`` and differencing them into class
probabilities. Provided so the "should this be ordinal regression?" question is
*reproducible* rather than asserted.

Empirically, on this synthetic dataset it does **not** beat the plain
multiclass RandomForest (see ``src.evaluate.compare_ordinal_vs_flat`` and the
README): the forest already respects the ordering (quadratic-weighted kappa
~0.73). It is kept as a documented, tested alternative, not the default model.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone


class OrdinalClassifier:
    """Frank & Hall (2001) ordinal meta-classifier over a binary base estimator.

    The base estimator must implement ``fit`` and ``predict_proba``. Inputs are
    expected to be already-numeric arrays (e.g. the output of the preprocessing
    ``ColumnTransformer``).
    """

    def __init__(self, base):
        self.base = base

    def fit(self, X, y) -> OrdinalClassifier:
        y = np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        self.clfs_: dict = {}
        for k in self.classes_[:-1]:
            binary_target = (y > k).astype(int)
            estimator = clone(self.base)
            estimator.fit(X, binary_target)
            self.clfs_[k] = estimator
        return self

    def predict_proba(self, X) -> np.ndarray:
        n = X.shape[0]
        k_classes = len(self.classes_)
        p_gt = {k: self.clfs_[k].predict_proba(X)[:, 1] for k in self.classes_[:-1]}

        proba = np.zeros((n, k_classes))
        for i in range(k_classes):
            if i == 0:
                proba[:, i] = 1.0 - p_gt[self.classes_[0]]
            elif i == k_classes - 1:
                proba[:, i] = p_gt[self.classes_[-2]]
            else:
                proba[:, i] = p_gt[self.classes_[i - 1]] - p_gt[self.classes_[i]]

        proba = np.clip(proba, 1e-9, None)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
