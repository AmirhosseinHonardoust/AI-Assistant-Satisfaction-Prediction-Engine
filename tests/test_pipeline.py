from __future__ import annotations

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLS, TARGET_COL
from src.data_prep import engineer_time_features, split_train_test
from src.features import build_pipeline
from src.generate_data import generate


def _prepared(n=400, seed=3):
    return engineer_time_features(generate(n=n, seed=seed))


def test_build_pipeline_fits_and_predicts():
    df = _prepared()
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    pipe.fit(df[FEATURE_COLS], df[TARGET_COL])
    preds = pipe.predict(df[FEATURE_COLS].head(10))
    assert len(preds) == 10
    assert set(preds) <= {1, 2, 3, 4, 5}


def test_split_is_stratified_and_complete():
    df = _prepared(n=500)
    train, test = split_train_test(df)
    assert len(train) + len(test) == len(df)
    # every class present in the test split (stratified)
    assert set(test[TARGET_COL].unique()) == set(df[TARGET_COL].unique())


def test_score_file_adds_prediction_columns(tmp_path, monkeypatch):
    import src.score_new_sessions as sns

    df = _prepared(n=300)
    pipe = build_pipeline()
    pipe.fit(df[FEATURE_COLS], df[TARGET_COL])
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipe, model_path)
    monkeypatch.setattr(sns, "load_model", lambda: joblib.load(model_path))

    raw = generate(n=20, seed=99).drop(columns=[TARGET_COL])
    in_csv = tmp_path / "sessions.csv"
    raw.to_csv(in_csv, index=False)

    out = sns.score_file(in_csv, tmp_path / "scored.csv")
    scored = pd.read_csv(out)
    assert "pred_satisfaction" in scored.columns
    prob_cols = [c for c in scored.columns if c.startswith("p_rating_")]
    assert len(prob_cols) == 5
    # probabilities sum to ~1 per row
    assert scored[prob_cols].sum(axis=1).round(3).eq(1.0).all()
