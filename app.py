import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, TARGET_COL, TIME_COL  # type: ignore


@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "satisfaction_pipeline.joblib"
    if not model_path.exists():
        st.error(
            f"Model not found at {model_path}. Run `python -m src.train_model` first."
        )
        st.stop()
    model = joblib.load(model_path)
    return model


@st.cache_data
def load_sample_data():
    test_path = PROCESSED_DATA_DIR / "sessions_test.csv"
    if not test_path.exists():
        st.error(
            f"Processed test data not found at {test_path}. "
            "Run `python -m src.data_prep` first."
        )
        st.stop()
    df = pd.read_csv(test_path)
    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if TIME_COL not in df.columns:
        return df
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df["hour_of_day"] = df[TIME_COL].dt.hour
    df["day_of_week"] = df[TIME_COL].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def score_sessions(model, df: pd.DataFrame) -> pd.DataFrame:
    # Ensure time features exist
    df_feat = engineer_time_features(df)
    df_features = df_feat.drop(columns=[TARGET_COL], errors="ignore")

    preds = model.predict(df_features)
    proba = model.predict_proba(df_features)

    df_scored = df_feat.copy()
    df_scored["pred_satisfaction"] = preds

    classes = model.classes_
    for i, c in enumerate(classes):
        df_scored[f"p_rating_{c}"] = proba[:, i]

    return df_scored


@st.cache_resource
def get_shap_explainer(_model):
    preprocessor = _model.named_steps["preprocess"]
    clf = _model.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    feature_names = preprocessor.get_feature_names_out()
    return explainer, preprocessor, feature_names


def plot_single_shap_bar(shap_values, feature_names, max_features: int = 10):
    shap_values = np.asarray(shap_values).reshape(-1)
    feature_names = np.asarray(feature_names)

    n_features = min(len(shap_values), len(feature_names))
    shap_values = shap_values[:n_features]
    feature_names = feature_names[:n_features]

    abs_vals = np.abs(shap_values)
    idx_sorted = np.argsort(abs_vals)[::-1][:max_features]

    selected_shap = shap_values[idx_sorted]
    selected_names = feature_names[idx_sorted]

    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(selected_names))

    ax.barh(y_pos, selected_shap)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(selected_names)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (impact on satisfaction prediction)")
    ax.set_title("Top feature contributions for this session")

    plt.tight_layout()
    return fig


def explain_single_session(model, df_scored: pd.DataFrame, row_idx: int):
    explainer, preprocessor, feature_names = get_shap_explainer(model)

    cols_to_drop = [TARGET_COL, "pred_satisfaction"]
    cols_to_drop.extend([c for c in df_scored.columns if c.startswith("p_rating_")])

    features_df = df_scored.drop(columns=[c for c in cols_to_drop if c in df_scored.columns])

    x_row = features_df.iloc[[row_idx]]
    x_transformed = preprocessor.transform(x_row)

    try:
        import scipy.sparse as sp

        if sp.issparse(x_transformed):
            x_for_shap = x_transformed.toarray()
        else:
            x_for_shap = x_transformed
    except ImportError:
        x_for_shap = x_transformed

    shap_vals = explainer.shap_values(x_for_shap)

    # Multiclass: shap_vals is list [class0,...]
    if isinstance(shap_vals, list):
        # choose class predicted for this row
        pred_class = df_scored.iloc[row_idx]["pred_satisfaction"]
        classes = model.classes_.tolist()
        class_index = classes.index(pred_class)
        shap_for_class = shap_vals[class_index][0]
    else:
        shap_for_class = shap_vals[0]

    fig = plot_single_shap_bar(shap_for_class, feature_names)
    return fig


def main():
    st.set_page_config(
        page_title="AI Assistant Satisfaction Prediction",
        layout="wide",
    )

    st.title("AI Assistant Satisfaction Prediction Engine")
    st.markdown(
        """
This dashboard uses a trained machine learning model to predict **user satisfaction**
with an AI assistant based on usage behavior, and explains which features drive each prediction.

- Upload a CSV of sessions or use the sample test set  
- See **satisfaction distribution** across devices, usage categories, and models  
- Explore **per-session predictions** and **SHAP explanations**
"""
    )

    model = load_model()

    st.sidebar.header("Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload sessions CSV", type=["csv"])

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success("Using uploaded data.")
    else:
        df_raw = load_sample_data()
        st.sidebar.info("Using sample test data from the project.")

    df_scored = score_sessions(model, df_raw)

    st.subheader("Overview")
    total_sessions = len(df_scored)
    true_available = TARGET_COL in df_scored.columns
    pred_distribution = df_scored["pred_satisfaction"].value_counts(normalize=True).sort_index()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total sessions", f"{total_sessions:,}")
    if true_available:
        true_distribution = (
            df_scored[TARGET_COL].value_counts(normalize=True).sort_index()
        )
        avg_true = df_scored[TARGET_COL].mean()
        col2.metric("Avg true satisfaction", f"{avg_true:.2f}/5")
    else:
        col2.metric("Avg true satisfaction", "Unknown")

    avg_pred = df_scored["pred_satisfaction"].mean()
    col3.metric("Avg predicted satisfaction", f"{avg_pred:.2f}/5")

    st.subheader("Satisfaction Distribution")

    c1, c2 = st.columns([2, 3])
    with c1:
        fig, ax = plt.subplots()
        idx = sorted(pred_distribution.index.tolist())
        pred_vals = [pred_distribution.get(i, 0) for i in idx]

        if true_available:
            true_distribution = (
                df_scored[TARGET_COL].value_counts(normalize=True).sort_index()
            )
            true_vals = [true_distribution.get(i, 0) for i in idx]
        else:
            true_vals = None

        width = 0.35
        x = np.arange(len(idx))

        if true_vals is not None:
            ax.bar(x - width / 2, true_vals, width, label="True")
            ax.bar(x + width / 2, pred_vals, width, label="Predicted")
        else:
            ax.bar(x, pred_vals, width, label="Predicted")

        ax.set_xticks(x)
        ax.set_xticklabels(idx)
        ax.set_ylabel("Proportion")
        ax.set_xlabel("Satisfaction rating")
        ax.set_title("Satisfaction Distribution")
        ax.legend()
        st.pyplot(fig)

    with c2:
        st.markdown("**Scored sessions (top 30)**")
        st.dataframe(df_scored.head(30))

    st.subheader("Segmented View")

    seg_col = st.selectbox(
        "Group by feature",
        options=["device", "usage_category", "assistant_model"],
    )

    if seg_col in df_scored.columns:
        grouped = df_scored.groupby(seg_col)["pred_satisfaction"].mean().sort_values(
            ascending=False
        )
        fig, ax = plt.subplots()
        ax.bar(grouped.index.astype(str), grouped.values)
        ax.set_ylabel("Avg predicted satisfaction")
        ax.set_title(f"Average predicted satisfaction by {seg_col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Explain a Single Session")
    if len(df_scored) == 0:
        st.warning("No data available.")
        return

    row_idx = st.number_input(
        "Row index to explain (0-based)", min_value=0, max_value=len(df_scored) - 1, value=0
    )

    row = df_scored.iloc[row_idx]
    st.markdown("**Selected session**")
    st.write(row.to_frame().T)

    st.markdown("**Model prediction**")
    st.write(
        f"Predicted satisfaction: **{row['pred_satisfaction']}**"
    )

    st.markdown("**Feature contribution (SHAP)**")
    with st.spinner("Computing SHAP values..."):
        fig_shap = explain_single_session(model, df_scored, row_idx)
        st.pyplot(fig_shap)


if __name__ == "__main__":
    main()
