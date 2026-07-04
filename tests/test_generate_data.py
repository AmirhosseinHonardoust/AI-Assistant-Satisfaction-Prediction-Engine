from __future__ import annotations

from src.config import CATEGORICAL_FEATURES, TARGET_COL
from src.generate_data import generate
from src.time_features import add_time_features


def test_generate_is_deterministic():
    a = generate(n=200, seed=123)
    b = generate(n=200, seed=123)
    assert a.equals(b)


def test_generate_schema_and_range():
    df = generate(n=300, seed=1)
    expected = {
        "timestamp",
        "device",
        "usage_category",
        "prompt_length",
        "session_length_minutes",
        TARGET_COL,
        "assistant_model",
        "tokens_used",
    }
    assert expected <= set(df.columns)
    assert df[TARGET_COL].between(1, 5).all()
    assert set(df[TARGET_COL].unique()) <= {1, 2, 3, 4, 5}


def test_signal_is_learnable_beats_majority_baseline():
    """A shallow tree on the generated data must clearly beat the majority
    baseline -- i.e. the designed signal is real, not noise."""
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    df = add_time_features(generate(n=600, seed=7))
    num = ["session_length_minutes", "is_weekend"]
    X = df[CATEGORICAL_FEATURES + num]
    y = df[TARGET_COL]

    pipe = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("c", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
                    remainder="passthrough",
                ),
            ),
            ("clf", DecisionTreeClassifier(max_depth=6, random_state=0)),
        ]
    )
    acc = cross_val_score(pipe, X, y, cv=3).mean()
    majority = y.value_counts(normalize=True).max()
    assert acc > majority + 0.08, f"acc={acc:.3f} not clearly above baseline={majority:.3f}"
    assert acc > 0.30  # comfortably above 5-class chance (~0.20)
