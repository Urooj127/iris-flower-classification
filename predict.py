"""
predict.py
----------
Command-line tool to predict the Iris species for a new flower measurement
using the saved model from 'models/best_model.pkl'.

Usage:
    python src/predict.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2

If no arguments are given, an interactive prompt is used instead.
"""

import argparse
import json
import os
import joblib
import numpy as np

MODELS_DIR = "models"


def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    meta_path = os.path.join(MODELS_DIR, "metadata.json")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError(
            "Trained model not found. Run 'python src/train_model.py' first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, scaler, metadata


def predict(sepal_length, sepal_width, petal_length, petal_width):
    model, scaler, metadata = load_artifacts()
    target_names = metadata["target_names"]

    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_scaled = scaler.transform(X)

    pred_idx = model.predict(X_scaled)[0]
    pred_species = target_names[pred_idx]

    result = {"prediction": pred_species}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        result["probabilities"] = {
            target_names[i]: round(float(p), 4) for i, p in enumerate(probs)
        }
    return result


def interactive():
    print("Enter flower measurements (in cm):")
    sepal_length = float(input("  Sepal length: "))
    sepal_width = float(input("  Sepal width : "))
    petal_length = float(input("  Petal length: "))
    petal_width = float(input("  Petal width : "))
    return sepal_length, sepal_width, petal_length, petal_width


def main():
    parser = argparse.ArgumentParser(description="Predict Iris species from flower measurements.")
    parser.add_argument("--sepal_length", type=float)
    parser.add_argument("--sepal_width", type=float)
    parser.add_argument("--petal_length", type=float)
    parser.add_argument("--petal_width", type=float)
    args = parser.parse_args()

    if None in (args.sepal_length, args.sepal_width, args.petal_length, args.petal_width):
        sepal_length, sepal_width, petal_length, petal_width = interactive()
    else:
        sepal_length, sepal_width = args.sepal_length, args.sepal_width
        petal_length, petal_width = args.petal_length, args.petal_width

    result = predict(sepal_length, sepal_width, petal_length, petal_width)

    print("\n" + "=" * 40)
    print(f"Predicted species: {result['prediction'].upper()}")
    if "probabilities" in result:
        print("\nClass probabilities:")
        for species, prob in result["probabilities"].items():
            print(f"  {species:12s}: {prob:.2%}")
    print("=" * 40)


if __name__ == "__main__":
    main()
