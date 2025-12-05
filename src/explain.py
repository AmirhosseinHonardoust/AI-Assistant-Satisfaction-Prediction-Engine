import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import scipy.sparse as sp

from .config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_COL, FIGURES_DIR


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


def compute_global_shap(model, X: pd.DataFrame, max_samples: int = 200):
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=0)
    else:
        X_sample = X

    preprocessor = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    X_transformed = preprocessor.transform(X_sample)

    if sp.issparse(X_transformed):
        X_for_shap = X_transformed.toarray()
    else:
        X_for_shap = X_transformed

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_for_shap)

    feature_names = preprocessor.get_feature_names_out()

    # For multiclass, shap_values is a list; average absolute impact across classes
    if isinstance(shap_values, list):
        # Concatenate along axis=0 (classes)
        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_abs = np.abs(shap_values)

    shap.summary_plot(
        shap_abs,
        X_for_shap,
        feature_names=feature_names,
        show=False,
    )

    shap_path = FIGURES_DIR / "shap_summary.png"
    plt.title("SHAP Summary - Satisfaction Model")
    plt.savefig(shap_path, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary to: {shap_path}")


def main():
    model, X_test, y_test = load_model_and_data()
    compute_global_shap(model, X_test)


if __name__ == "__main__":
    main()
