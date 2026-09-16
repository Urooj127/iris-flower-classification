"""
test_project.py
----------------
Basic automated tests covering data loading, model training artifacts,
and the prediction pipeline.

Run:
    pytest tests/
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_exploration import load_data  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")


def test_data_shape():
    df = load_data()
    assert df.shape[0] == 150
    assert df.shape[1] == 6


def test_no_missing_values():
    df = load_data()
    assert df.isnull().sum().sum() == 0


def test_three_classes():
    df = load_data()
    assert df["species"].nunique() == 3
    assert set(df["species"].unique()) == {"setosa", "versicolor", "virginica"}


def test_balanced_classes():
    df = load_data()
    counts = df["species"].value_counts()
    assert (counts == 50).all()


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")),
    reason="Model not trained yet - run src/train_model.py first",
)
def test_model_prediction_shape():
    import joblib

    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    X = np.array([[5.1, 3.5, 1.4, 0.2]])  # classic setosa sample
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    assert pred in (0, 1, 2)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "metadata.json")),
    reason="Model not trained yet - run src/train_model.py first",
)
def test_model_accuracy_above_threshold():
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    best_name = metadata["best_model_name"]
    best_acc = metadata["all_results"][best_name]["test_accuracy"]
    assert best_acc >= 0.85  # Iris is an easy dataset; a sane model should clear this easily
