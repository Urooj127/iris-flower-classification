"""
data_exploration.py
--------------------
Loads the Iris dataset, performs basic exploratory data analysis (EDA),
and saves a clean CSV copy of the dataset for reference.

Run:
    python src/data_exploration.py
"""

import os
import pandas as pd
from sklearn.datasets import load_iris


def load_data():
    """Load the Iris dataset from scikit-learn and return it as a DataFrame."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.rename(columns={"target": "species_id"}, inplace=True)
    df["species"] = df["species_id"].map(dict(enumerate(iris.target_names)))
    return df


def explore(df: pd.DataFrame):
    print("=" * 60)
    print("IRIS DATASET - BASIC EXPLORATION")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nStatistical summary:")
    print(df.describe())

    print("\nClass distribution:")
    print(df["species"].value_counts())

    print("\nCorrelation matrix (numeric features):")
    numeric_cols = df.select_dtypes(include="number").drop(columns=["species_id"])
    print(numeric_cols.corr())


def main():
    df = load_data()
    explore(df)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "iris.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved dataset to '{out_path}'")


if __name__ == "__main__":
    main()
