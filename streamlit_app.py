"""
streamlit_app.py
-----------------
Interactive web app for the Iris Flower Classification project.

Run locally:
    streamlit run streamlit_app.py
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from data_exploration import load_data  # noqa: E402

ROOT = os.path.dirname(__file__)
MODELS_DIR = ROOT

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return model, scaler, metadata


@st.cache_data
def get_data():
    return load_data()


def predict_species(model, scaler, metadata, sl, sw, pl, pw):
    X = np.array([[sl, sw, pl, pw]])
    X_scaled = scaler.transform(X)
    pred_idx = model.predict(X_scaled)[0]
    target_names = metadata["target_names"]
    pred_species = target_names[pred_idx]
    probs = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_scaled)[0]
        probs = {target_names[i]: float(p[i]) for i in range(len(p))}
    return pred_species, probs


SPECIES_INFO = {
    "setosa": "🌱 Iris Setosa — smallest petals, easiest to distinguish from the other two species.",
    "versicolor": "🌿 Iris Versicolor — medium-sized petals, overlaps a bit with Virginica.",
    "virginica": "🌳 Iris Virginica — largest petals on average.",
}


def main():
    st.title("🌸 Iris Flower Dataset — Analysis & Classification")
    st.caption("A beginner-friendly end-to-end ML project: EDA, model comparison, and live prediction.")

    tab_predict, tab_eda, tab_models, tab_about = st.tabs(
        ["🔮 Predict", "📊 Explore Data", "🏆 Model Comparison", "ℹ️ About"]
    )

    # ---------------- Predict tab ----------------
    with tab_predict:
        try:
            model, scaler, metadata = load_artifacts()
        except FileNotFoundError:
            st.error("Trained model not found. Please run `python train_model.py` first.")
            return

        st.subheader("Enter flower measurements")
        col1, col2 = st.columns(2)
        with col1:
            sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
            sw = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
        with col2:
            pl = st.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
            pw = st.slider("Petal width (cm)", 0.1, 2.6, 1.3, 0.1)

        if st.button("Predict species", type="primary"):
            pred, probs = predict_species(model, scaler, metadata, sl, sw, pl, pw)
            st.success(f"Predicted species: **{pred.upper()}**")
            st.write(SPECIES_INFO.get(pred, ""))

            if probs:
                prob_df = pd.DataFrame({"species": list(probs.keys()), "probability": list(probs.values())})
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(data=prob_df, x="probability", y="species", hue="species", palette="Set2", legend=False, ax=ax)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.set_ylabel("")
                for i, v in enumerate(prob_df["probability"]):
                    ax.text(v + 0.01, i, f"{v:.1%}", va="center")
                st.pyplot(fig)

        st.info(f"Model in use: **{metadata['best_model_name']}**")

    # ---------------- EDA tab ----------------
    with tab_eda:
        df = get_data()
        st.subheader("Dataset preview")
        st.dataframe(df.drop(columns=["species_id"]).head(10), width="stretch")

        st.subheader("Statistical summary")
        st.dataframe(df.describe(), width="stretch")

        st.subheader("Class distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(data=df, x="species", hue="species", palette="Set2", legend=False, ax=ax)
        st.pyplot(fig)

        st.subheader("Feature relationships")
        feature_x = st.selectbox("X-axis feature", [c for c in df.columns if "(cm)" in c], index=2)
        feature_y = st.selectbox("Y-axis feature", [c for c in df.columns if "(cm)" in c], index=3)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x=feature_x, y=feature_y, hue="species", palette="Set2", s=60, ax=ax)
        st.pyplot(fig)

    # ---------------- Model comparison tab ----------------
    with tab_models:
        try:
            _, _, metadata = load_artifacts()
        except FileNotFoundError:
            st.warning("Train the models first with `python train_model.py`.")
        else:
            st.subheader("Test accuracy by model")
            results = metadata["all_results"]
            results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
            st.dataframe(results_df, width="stretch")

            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=results_df, x="test_accuracy", y="model", hue="model", palette="viridis", legend=False, ax=ax)
            ax.set_xlim(0, 1.05)
            st.pyplot(fig)

            st.success(f"Best model selected: **{metadata['best_model_name']}**")

    # ---------------- About tab ----------------
    with tab_about:
        st.markdown(
            """
            ### About this project
            This app classifies iris flowers into **setosa**, **versicolor**, or **virginica**
            based on four measurements: sepal length, sepal width, petal length, and petal width.

            It is built as a beginner-level, end-to-end machine learning project covering:
            - Exploratory Data Analysis (EDA)
            - Training and comparing multiple classic ML models
            - Deploying an interactive prediction tool

            See the project `README.md` for full documentation, architecture, and results.
            """
        )


if __name__ == "__main__":
    main()
