# app/streamlit_app.py
import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# allow importing ../backend
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pipeline import (
    extract_features_from_folder,
    scale_features,
    correlations,
    select_kbest,
    train_random_forest,
    train_xgboost,
)

st.set_page_config(page_title="Chatter Detection UI", layout="wide")

st.title("🧰 Chatter Detection")
st.caption("Upload/point to a folder of CSV/XLSX files with columns: FZ, DOC, SPEED, FEED, CHATTER")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    folder_path = st.text_input("Data folder (absolute path)", value=os.path.abspath(os.path.join("..", "data")))
    segment_size = st.number_input("Segment size", min_value=128, max_value=8192, value=1024, step=128)
    step_size = st.number_input("Step size", min_value=32, max_value=4096, value=256, step=32)
    sampling_rate = st.number_input("Sampling rate (Hz)", min_value=100, max_value=100_000, value=10005, step=5)
    preview = st.checkbox("Show preview plots (first few files)", value=True)
    k_features = st.slider("K best features", min_value=3, max_value=30, value=6, step=1)
    model_name = st.selectbox("Model", ["RandomForest", "XGBoost"])
    st.subheader("Model hyperparameters")

    if model_name == "RandomForest":
        n_estimators = st.slider("n_estimators", 50, 500, 200, step=50)
        max_depth = st.selectbox("max_depth", [None, 10, 20, 30], index=0)
        min_samples_split = st.selectbox("min_samples_split", [2, 5, 10], index=0)
        params = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    else:
        n_estimators = st.slider("n_estimators", 50, 500, 100, step=50)
        max_depth = st.slider("max_depth", 2, 12, 5, step=1)
        learning_rate = st.selectbox("learning_rate", [0.05, 0.1, 0.2], index=1)
        params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", 0, 10_000, 42)

# -----------------------------
# Caching heavy steps
# -----------------------------
@st.cache_data(show_spinner=True)
def _cached_extract(folder_path, segment_size, step_size, sampling_rate, preview):
    df, figs = extract_features_from_folder(
        folder_path=folder_path,
        segment_size=segment_size,
        step_size=step_size,
        sampling_rate=sampling_rate,
        preview_plots=preview,
        preview_limit=2
    )
    return df, figs

@st.cache_data(show_spinner=True)
def _cached_scale(features_df):
    return scale_features(features_df)

@st.cache_data(show_spinner=True)
def _cached_select_kbest(X_scaled_df, y, k):
    return select_kbest(X_scaled_df, y, k)

# -----------------------------
# Buttons
# -----------------------------
colA, colB = st.columns([1,1])
with colA:
    run_extract = st.button("🔎 Extract Features", type="primary")
with colB:
    run_train = st.button("🚀 Train Model", disabled=("features_df" not in st.session_state))

# -----------------------------
# Step 1: Extract
# -----------------------------
if run_extract:
    if not os.path.isdir(folder_path):
        st.error("Folder not found. Please provide a valid path.")
    else:
        with st.spinner("Extracting features..."):
            features_df, preview_figs = _cached_extract(folder_path, segment_size, step_size, sampling_rate, preview)
        if features_df.empty:
            st.warning("No samples extracted. Check your files/columns.")
        else:
            st.session_state["features_df"] = features_df
            st.success(f"Feature extraction complete: {features_df.shape[0]} samples, {features_df.shape[1]} columns")

# Show preview
if "features_df" in st.session_state:
    st.subheader("Features preview")
    st.dataframe(st.session_state["features_df"].head(), use_container_width=True)

    # download features
    csv_bytes = st.session_state["features_df"].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download features CSV", data=csv_bytes, file_name="features.csv", mime="text/csv")

    # preview plots
    _, preview_figs = _cached_extract(folder_path, segment_size, step_size, sampling_rate, preview)
    if preview and preview_figs:
        st.subheader("Signal previews")
        for fig in preview_figs:
            st.pyplot(fig, clear_figure=False)

    # correlations
    st.subheader("Correlation heatmap")
    X_scaled_df, y, _ = _cached_scale(st.session_state["features_df"])
    corr = correlations(X_scaled_df, y)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, cbar=True, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

    # chatter-only correlations
    if "chatter" in corr.columns:
        st.subheader("Correlation with 'chatter'")
        col = corr[["chatter"]].sort_values("chatter", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(3, 10))
        sns.heatmap(col, cmap="coolwarm", annot=True, fmt=".2f", cbar=False, ax=ax2)
        st.pyplot(fig2)

# -----------------------------
# Step 2: Train
# -----------------------------
if run_train:
    if "features_df" not in st.session_state or st.session_state["features_df"].empty:
        st.error("Run feature extraction first.")
    else:
        with st.spinner("Preparing data..."):
            X_scaled_df, y, scaler = _cached_scale(st.session_state["features_df"])
            X_sel_df, cols, selector = _cached_select_kbest(X_scaled_df, y, k_features)

        st.info(f"Selected top-{len(cols)} features: {', '.join(cols)}")

        with st.spinner(f"Training {model_name}..."):
            if model_name == "RandomForest":
                model, splits, metrics = train_random_forest(X_sel_df, y, test_size=test_size,
                                                            random_state=random_state, params=params)
            else:
                model, splits, metrics = train_xgboost(X_sel_df, y, test_size=test_size,
                                                       random_state=random_state, params=params)

        st.subheader("Results")
        c1, c2 = st.columns([1,1])
        with c1:
            st.write(f"**Train accuracy:** {metrics['train_acc']:.4f}")
            st.write(f"**Test accuracy:** {metrics['test_acc']:.4f}")
        with c2:
            st.text("Classification Report")
            st.code(metrics["report"])

        fig_cm, ax_cm = plt.subplots(figsize=(4,4))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cbar=False, ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        # allow download of model + scaler + selector
        os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
        bundle = dict(model=model, scaler=scaler, selector=selector, selected_columns=list(cols))
        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button("💾 Download trained bundle (.joblib)",
                           data=buf.getvalue(),
                           file_name="chatter_model_bundle.joblib",
                           mime="application/octet-stream")
