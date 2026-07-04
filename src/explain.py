from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import shap

from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, TARGET_COL
from .shap_utils import mean_abs_shap_across_classes


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


def compute_global_shap(model, X: pd.DataFrame, max_samples: int = 200) -> None:
    X_sample = X.sample(max_samples, random_state=0) if len(X) > max_samples else X

    preprocessor = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    X_transformed = preprocessor.transform(X_sample)
    X_for_shap = X_transformed.toarray() if sp.issparse(X_transformed) else X_transformed

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_for_shap)
    feature_names = preprocessor.get_feature_names_out()

    shap_abs = mean_abs_shap_across_classes(shap_values)

    shap.summary_plot(shap_abs, X_for_shap, feature_names=feature_names, show=False)

    shap_path = FIGURES_DIR / "shap_summary.png"
    plt.title("SHAP Summary - Satisfaction Model")
    plt.savefig(shap_path, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary to: {shap_path}")


def main() -> None:
    model, X_test, _ = load_model_and_data()
    compute_global_shap(model, X_test)


if __name__ == "__main__":
    main()
