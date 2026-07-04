from __future__ import annotations

import numpy as np

from src.shap_utils import mean_abs_shap_across_classes, shap_row_for_class

N_SAMPLES, N_FEATURES, N_CLASSES = 4, 6, 3


def _list_form():
    rng = np.random.default_rng(0)
    return [rng.normal(size=(N_SAMPLES, N_FEATURES)) for _ in range(N_CLASSES)]


def _ndarray_form(list_form):
    # modern SHAP: (n_samples, n_features, n_classes)
    return np.stack(list_form, axis=-1)


def test_mean_abs_shapes_match_across_conventions():
    lst = _list_form()
    arr = _ndarray_form(lst)

    out_list = mean_abs_shap_across_classes(lst)
    out_arr = mean_abs_shap_across_classes(arr)

    assert out_list.shape == (N_SAMPLES, N_FEATURES)
    assert out_arr.shape == (N_SAMPLES, N_FEATURES)
    np.testing.assert_allclose(out_list, out_arr)


def test_shap_row_for_class_matches_across_conventions():
    lst = _list_form()
    arr = _ndarray_form(lst)

    row, cls = 2, 1
    from_list = shap_row_for_class(lst, row=row, class_index=cls)
    from_arr = shap_row_for_class(arr, row=row, class_index=cls)

    assert from_list.shape == (N_FEATURES,)
    assert from_arr.shape == (N_FEATURES,)
    # Both must equal the raw per-class, per-row slice -- this is exactly the
    # mapping the old code got wrong for the ndarray convention.
    np.testing.assert_allclose(from_list, lst[cls][row])
    np.testing.assert_allclose(from_arr, lst[cls][row])
