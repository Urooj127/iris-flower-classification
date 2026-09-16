"""
visualize.py
------------
Generates exploratory visualizations for the Iris dataset and saves them
as PNG images inside the 'images/' folder.

Run:
    python src/visualize.py
"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

from data_exploration import load_data

sns.set_theme(style="whitegrid")
IMG_DIR = "images"


def save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="species", hue="species", palette="Set2", legend=False, ax=ax)
    ax.set_title("Class Distribution")
    save(fig, "01_class_distribution.png")


def plot_pairplot(df):
    g = sns.pairplot(df.drop(columns=["species_id"]), hue="species", palette="Set2", diag_kind="hist")
    g.fig.suptitle("Pairwise Feature Relationships", y=1.02)
    g.fig.savefig(os.path.join(IMG_DIR, "02_pairplot.png"), bbox_inches="tight", dpi=150)
    plt.close(g.fig)
    print("Saved: images/02_pairplot.png")


def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include="number").drop(columns=["species_id"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    save(fig, "03_correlation_heatmap.png")


def plot_boxplots(df):
    features = [c for c in df.columns if "(cm)" in c]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, feature in zip(axes.flatten(), features):
        sns.boxplot(data=df, x="species", y=feature, hue="species", palette="Set2", legend=False, ax=ax)
        ax.set_title(feature)
    fig.suptitle("Feature Distributions by Species (Boxplots)")
    fig.tight_layout()
    save(fig, "04_boxplots.png")


def plot_violin(df):
    features = [c for c in df.columns if "(cm)" in c]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, feature in zip(axes.flatten(), features):
        sns.violinplot(data=df, x="species", y=feature, hue="species", palette="Set2", legend=False, ax=ax)
        ax.set_title(feature)
    fig.suptitle("Feature Distributions by Species (Violin Plots)")
    fig.tight_layout()
    save(fig, "05_violinplots.png")


def main():
    df = load_data()
    plot_class_distribution(df)
    plot_pairplot(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    plot_violin(df)
    print("\nAll visualizations generated successfully.")


if __name__ == "__main__":
    main()
