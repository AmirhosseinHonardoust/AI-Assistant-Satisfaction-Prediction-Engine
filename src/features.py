from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model() -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return model


def build_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()
    model = build_model()

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", model),
        ]
    )
    return pipeline
