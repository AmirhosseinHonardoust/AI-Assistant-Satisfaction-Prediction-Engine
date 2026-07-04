from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.ordinal import OrdinalClassifier


def _ordered_data(n=300, seed=0):
    """Feature correlated with an ordered 1-5 label."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    edges = np.quantile(x[:, 0], [0.2, 0.4, 0.6, 0.8])
    y = np.digitize(x[:, 0], edges) + 1
    return x, y


def test_fit_predict_shapes_and_classes():
    X, y = _ordered_data()
    clf = OrdinalClassifier(DecisionTreeClassifier(max_depth=4, random_state=0)).fit(X, y)
    assert list(clf.classes_) == [1, 2, 3, 4, 5]
    preds = clf.predict(X[:20])
    assert preds.shape == (20,)
    assert set(preds).issubset({1, 2, 3, 4, 5})


def test_proba_rows_sum_to_one():
    X, y = _ordered_data()
    clf = OrdinalClassifier(DecisionTreeClassifier(max_depth=4, random_state=0)).fit(X, y)
    proba = clf.predict_proba(X[:50])
    assert proba.shape == (50, 5)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all()


def test_learns_ordering():
    X, y = _ordered_data(n=600)
    clf = OrdinalClassifier(DecisionTreeClassifier(max_depth=6, random_state=0)).fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.4  # comfortably beats 5-class chance on this separable signal
