"""
train_model.py
---------------
Trains several classic ML classifiers on the Iris dataset, compares their
performance using cross-validation and a held-out test set, and saves the
best-performing model (plus the fitted scaler) to the 'models/' folder.

Run:
    python src/train_model.py
"""

import os
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_exploration import load_data

RANDOM_STATE = 42
MODELS_DIR = "models"
IMG_DIR = "images"


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
    }


def main():
    df = load_data()
    feature_cols = [c for c in df.columns if "(cm)" in c]
    X = df[feature_cols].values
    y = df["species_id"].values
    target_names = sorted(df["species"].unique(), key=lambda s: df.loc[df["species"] == s, "species_id"].iloc[0])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    fitted_models = {}

    print("=" * 60)
    print("MODEL TRAINING & COMPARISON (80/20 train-test split, 5-fold CV)")
    print("=" * 60)

    for name, model in get_models().items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        results[name] = {
            "test_accuracy": round(float(test_acc), 4),
            "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
        }
        fitted_models[name] = model

        print(f"\n{name}")
        print(f"  Test Accuracy      : {test_acc:.4f}")
        print(f"  CV Accuracy (mean) : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Pick best model by test accuracy (tie-break by CV mean)
    best_name = max(results, key=lambda n: (results[n]["test_accuracy"], results[n]["cv_mean_accuracy"]))
    best_model = fitted_models[best_name]

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}")
    print("=" * 60)
    y_pred_best = best_model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=target_names))

    # Save comparison bar chart
    os.makedirs(IMG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    accs = [results[n]["test_accuracy"] for n in names]
    sns.barplot(x=accs, y=names, hue=names, palette="viridis", legend=False, ax=ax)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Model Comparison - Test Accuracy")
    for i, v in enumerate(accs):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "06_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {IMG_DIR}/06_model_comparison.png")

    # Save confusion matrix for the best model
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "07_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {IMG_DIR}/07_confusion_matrix.png")

    # Persist best model + scaler + metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    metadata = {
        "best_model_name": best_name,
        "feature_columns": feature_cols,
        "target_names": list(target_names),
        "all_results": results,
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved trained model to '{MODELS_DIR}/best_model.pkl'")
    print(f"Saved scaler to '{MODELS_DIR}/scaler.pkl'")
    print(f"Saved metadata to '{MODELS_DIR}/metadata.json'")


if __name__ == "__main__":
    main()
