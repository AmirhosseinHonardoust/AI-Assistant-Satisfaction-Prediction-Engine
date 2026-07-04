from __future__ import annotations

import numpy as np

from src.model_metrics import (
    calibration_curve_points,
    expected_calibration_error,
    multiclass_log_loss,
    ordinal_metrics,
)


def test_ordinal_metrics_perfect_prediction():
    y = [1, 2, 3, 4, 5]
    m = ordinal_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["quadratic_weighted_kappa"] == 1.0


def test_mae_penalises_distance():
    # off-by-one vs off-by-three: MAE must reflect the larger error
    near = ordinal_metrics([3, 3], [2, 4])["mae"]
    far = ordinal_metrics([3, 3], [1, 5])["mae"]
    assert far > near


def test_ece_zero_when_confidence_matches_accuracy():
    # 100%-confident and always correct -> perfectly calibrated
    classes = np.array([0, 1])
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1, 0])
    assert expected_calibration_error(y, proba, classes) == 0.0


def test_ece_detects_overconfidence():
    classes = np.array([0, 1])
    # always 90% confident on class 1 but only right half the time
    proba = np.array([[0.1, 0.9]] * 4)
    y = np.array([1, 0, 1, 0])
    ece = expected_calibration_error(y, proba, classes)
    assert 0.35 < ece < 0.45  # |0.5 acc - 0.9 conf| = 0.4


def test_calibration_curve_and_log_loss_shapes():
    rng = np.random.default_rng(0)
    classes = np.array([0, 1, 2])
    proba = rng.dirichlet(np.ones(3), size=50)
    y = proba.argmax(axis=1)
    conf, acc, counts = calibration_curve_points(y, proba, classes)
    assert len(conf) == len(acc) == len(counts)
    assert multiclass_log_loss(y, proba, classes) >= 0.0
