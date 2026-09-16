# 🌸 Iris Flower Dataset — Analysis & Classification

A beginner-friendly, end-to-end machine learning project that explores the classic
**Iris flower dataset**, compares several classification algorithms, and ships an
interactive **Streamlit web app** that predicts a flower's species from its
measurements in real time.

> Status: ✅ Complete | Level: Beginner | Domain: Supervised Learning / Classification

---

## Table of Contents
1. [Project Overview & Objectives](#1-project-overview--objectives)
2. [Features & Technologies Used](#2-features--technologies-used)
3. [Installation / Setup Instructions](#3-installation--setup-instructions)
4. [Usage & Screenshots](#4-usage--screenshots)
5. [System Workflow / Architecture](#5-system-workflow--architecture)
6. [Testing / Results](#6-testing--results)
7. [Challenges & Future Improvements](#7-challenges--future-improvements)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Project Overview & Objectives

The **Iris dataset** (Fisher, 1936) is one of the most famous datasets in machine
learning: 150 flower samples from three species — *Iris setosa*, *Iris versicolor*,
and *Iris virginica* — each described by four measurements (sepal length, sepal
width, petal length, petal width).

### Objectives
- Perform thorough **exploratory data analysis (EDA)** to understand feature
  distributions and relationships.
- Train and fairly **compare multiple classification algorithms**.
- Select and persist the **best-performing model**.
- Build a simple **CLI tool** and an **interactive web app** for making live
  predictions on new flower measurements.
- Package everything with clean, well-documented, reproducible code — suitable
  as a first "real" data science / ML portfolio project.

### Who this is for
Beginners learning the standard ML workflow: load data → explore → preprocess →
train → evaluate → deploy.

---

## 2. Features & Technologies Used

### Features
- 📊 Automated EDA: summary statistics, class balance, correlation heatmap,
  pairplot, boxplots, violin plots.
- 🤖 Six classifiers trained and compared: Logistic Regression, K-Nearest
  Neighbors, Support Vector Machine, Decision Tree, Random Forest, Naive Bayes.
- 🏆 Automatic selection of the best model based on held-out test accuracy
  (tie-broken by cross-validation score).
- 💾 Model persistence (`joblib`) so the app doesn't retrain on every run.
- 🖥️ CLI prediction tool (`src/predict.py`) — pass measurements as arguments or
  enter them interactively.
- 🌐 Interactive **Streamlit** web app with:
  - Live prediction with sliders + probability chart
  - In-browser dataset explorer
  - Model comparison dashboard
- ✅ Automated test suite (`pytest`) covering data integrity and the prediction
  pipeline.

### Technologies
| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Data handling | pandas, NumPy |
| ML | scikit-learn |
| Visualization | matplotlib, seaborn |
| Web app | Streamlit |
| Model persistence | joblib |
| Testing | pytest |
| Version control / deployment | Git, GitHub, Streamlit Community Cloud |

---

## 3. Installation / Setup Instructions

### Prerequisites
- Python 3.9 or higher
- `pip` (or `conda`)
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/iris-flower-classification.git
cd iris-flower-classification

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline (in order)
python src/data_exploration.py   # explores the data, saves data/iris.csv
python src/visualize.py          # saves plots to images/
python src/train_model.py        # trains models, saves the best one to models/

# 5. Make a prediction from the command line
python src/predict.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2

# 6. Launch the interactive web app
streamlit run app/streamlit_app.py
```

The app will open automatically at `http://localhost:8501`.

### Running tests
```bash
pytest tests/ -v
```

---

## 4. Usage & Screenshots

### Command-line prediction
```text
$ python src/predict.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2

========================================
Predicted species: SETOSA

Class probabilities:
  setosa      : 97.21%
  versicolor  : 1.60%
  virginica   : 1.19%
========================================
```

### Web app
Run `streamlit run app/streamlit_app.py` and use the **Predict** tab to move the
sliders for sepal/petal length & width — the predicted species and class
probabilities update instantly. Other tabs let you explore the raw data and
compare model performance visually.

### Generated visualizations (`images/`)
| File | Description |
|---|---|
| `01_class_distribution.png` | Bar chart confirming the dataset is perfectly balanced (50 samples/class) |
| `02_pairplot.png` | Pairwise scatter plots of all four features, colored by species |
| `03_correlation_heatmap.png` | Correlation matrix of the four numeric features |
| `04_boxplots.png` | Boxplots of each feature grouped by species |
| `05_violinplots.png` | Violin plots showing feature distribution shape per species |
| `06_model_comparison.png` | Bar chart comparing test accuracy across all six models |
| `07_confusion_matrix.png` | Confusion matrix for the best-performing model (SVM) |

*(Screenshots are generated automatically the first time you run the scripts above — see the `images/` folder after running `visualize.py` and `train_model.py`.)*

---

## 5. System Workflow / Architecture

```
┌─────────────────────┐
│  sklearn.datasets    │
│  load_iris()         │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐        ┌─────────────────────┐
│ data_exploration.py  │──────▶│   data/iris.csv       │
│ (EDA + cleaning)      │        └─────────────────────┘
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐        ┌─────────────────────┐
│ visualize.py          │──────▶│   images/*.png         │
│ (charts & plots)       │        └─────────────────────┘
└──────────┬───────────┘
           │
           ▼
┌───────────────────────────────────────────┐
│ train_model.py                              │
│  • train/test split (80/20, stratified)      │
│  • StandardScaler                            │
│  • Train 6 classifiers                       │
│  • 5-fold cross-validation                   │
│  • Pick best model by test accuracy          │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  models/               │
│   best_model.pkl        │
│   scaler.pkl             │
│   metadata.json           │
└──────────┬───────────┘
           │
   ┌───────┴────────┐
   ▼                 ▼
┌───────────┐   ┌─────────────────────┐
│ predict.py │   │ app/streamlit_app.py │
│ (CLI tool)  │   │ (interactive web UI)  │
└───────────┘   └─────────────────────┘
```

### Repository structure
```
iris-flower-classification/
├── app/
│   └── streamlit_app.py       # Interactive web application
├── src/
│   ├── data_exploration.py    # Load + explore the dataset
│   ├── visualize.py           # Generate EDA plots
│   ├── train_model.py         # Train, compare, and save models
│   └── predict.py             # CLI prediction tool
├── tests/
│   └── test_project.py        # Automated tests (pytest)
├── data/
│   └── iris.csv                # Generated dataset export
├── images/                     # Generated charts (EDA + results)
├── models/                     # Saved model, scaler, metadata (generated)
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## 6. Testing / Results

### Automated tests
6 tests in `tests/test_project.py`, all passing:
- Dataset shape and column integrity
- No missing values
- Correct class labels and perfect class balance (50/50/50)
- Model produces a valid prediction
- Best model clears an 85% accuracy threshold

```bash
$ pytest tests/ -v
tests/test_project.py::test_data_shape PASSED
tests/test_project.py::test_no_missing_values PASSED
tests/test_project.py::test_three_classes PASSED
tests/test_project.py::test_balanced_classes PASSED
tests/test_project.py::test_model_prediction_shape PASSED
tests/test_project.py::test_model_accuracy_above_threshold PASSED
======================= 6 passed =======================
```

### Model comparison results
(80/20 stratified train-test split, 5-fold cross-validation on the training set)

| Model | Test Accuracy | CV Mean Accuracy | CV Std Dev |
|---|---|---|---|
| Logistic Regression | 0.9333 | 0.9583 | 0.0264 |
| K-Nearest Neighbors | 0.9333 | 0.9667 | 0.0312 |
| **Support Vector Machine** | **0.9667** | **0.9667** | **0.0312** |
| Decision Tree | 0.9333 | 0.9417 | 0.0204 |
| Random Forest | 0.9000 | 0.9500 | 0.0167 |
| Naive Bayes | 0.9667 | 0.9583 | 0.0264 |

**Best model: Support Vector Machine (RBF kernel)** — selected automatically by
the training script based on test accuracy.

### Classification report (best model, test set)
```
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

Only 1 of 30 test samples was misclassified (a versicolor predicted as
virginica) — consistent with the known fact that these two species have some
overlap in petal measurements, while setosa is linearly separable from the
other two.

**Note on exact numbers:** results were generated with a fixed `random_state=42`,
so re-running the pipeline as-is will reproduce these exact figures. Removing or
changing the random seed will produce slightly different (but similarly high)
scores, since the Iris dataset is small (150 samples) and easy to model well.

---

## 7. Challenges & Future Improvements

### Challenges encountered
- **Versicolor / virginica overlap**: these two species overlap somewhat in
  petal length and width, which is the main source of the ~3% test error across
  most models — setosa itself is always perfectly separated.
- **Small dataset size (150 rows)**: makes train/test splits and cross-validation
  scores somewhat sensitive to the random seed; addressed by using stratified
  splitting and 5-fold CV rather than relying on a single split.
- **Feature scaling**: models like KNN and SVM are distance-based and need
  standardized inputs; this was handled with `StandardScaler`, fit only on the
  training set to avoid data leakage.

### Future improvements
- Add hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`) for each model.
- Add more advanced models (Gradient Boosting, XGBoost) for comparison.
- Add SHAP or permutation feature-importance explanations to the app.
- Containerize with Docker for easier deployment.
- Add a REST API (FastAPI/Flask) alongside the Streamlit UI for programmatic access.
- Persist prediction history and allow CSV batch uploads for bulk predictions.
- Add CI (GitHub Actions) to run `pytest` automatically on every push.

---

## 8. Conclusion

This project walks through the complete, classic machine learning workflow on a
well-known dataset: loading and cleaning data, exploratory analysis, training
and fairly comparing multiple algorithms, and shipping a usable interactive
application. Despite its simplicity, the Iris dataset is an excellent teaching
tool — it makes concepts like class separability, overfitting, cross-validation,
and feature scaling tangible with a small, fast, and fully reproducible example.
The resulting Support Vector Machine model reaches **96.7% test accuracy**,
and the project is structured so every stage (EDA, training, inference, UI) is
a separate, readable, well-tested script.

---

## 9. References

- Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems.*
  Annals of Eugenics, 7(2), 179–188.
- Iris dataset via `sklearn.datasets.load_iris` — https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset
- UCI Machine Learning Repository — Iris Data Set — https://archive.ics.uci.edu/dataset/53/iris
- scikit-learn documentation — https://scikit-learn.org/stable/
- Streamlit documentation — https://docs.streamlit.io/
- seaborn documentation — https://seaborn.pydata.org/

---

## License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
