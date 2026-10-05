<div align="center">

# 🧪 Feature Engineering in Machine Learning
### A hands-on, end-to-end guide — from raw data to production-ready pipelines

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C8CBF)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

![Modules](https://img.shields.io/badge/modules-13-blue) ![Notebooks](https://img.shields.io/badge/notebooks-34-orange) ![Status](https://img.shields.io/badge/status-actively%20maintained-brightgreen)

</div>

---

## 📌 Overview

This repository is a **practical, production-oriented reference for feature engineering** — covering fundamental to advanced techniques across missing-data imputation, encoding, scaling, outlier handling, transformations, dimensionality reduction, and reusable scikit-learn pipelines.

Every notebook pairs a **concept** with a **reproducible implementation**, compares approaches (pandas vs. scikit-learn, from-scratch vs. library), and **measures impact** with visual diagnostics and model metrics — so you can see *why* a technique works, not just *how* to call it.

> 💡 **Who is this for?** Data scientists and ML engineers who want a clean, runnable cookbook; students learning the preprocessing stage of the ML lifecycle; and reviewers evaluating hands-on ML fundamentals.

---

## 📑 Table of Contents

- [Repository Architecture](#-repository-architecture)
- [Module Breakdown](#-module-breakdown)
- [Dataset Setup Guide](#-dataset-setup-guide)
- [Installation & Execution](#-installation--execution)
- [Key Methodological Highlights](#-key-methodological-highlights)
- [Suggested Learning Path](#-suggested-learning-path)
- [Author & Connect](#-author--connect)

---

## 🗂️ Repository Architecture

```text
Feature-Engineering/
├── Column Transformer/                              # 3 notebooks
├── Encoding numerical features/                     # 2 notebooks
├── Feature Construction and Feature Splitting/      # 1 notebook
├── Feature Scaling - Normalization/                 # 1 notebook
├── Feature Scaling - Standardization/               # 1 notebook
├── Handling Categorical Variables/
│   ├── Label and Ordinal Encoding/
│   └── One-Hot Encoding/
├── Handling Date and Time variables/                # 1 notebook
├── Handling Missing Data/
│   ├── Complete Case Analysis - dropping/
│   └── Imputation/
│       ├── Numerical(Mean, Median, Aribitrary, EOD)/
│       ├── Categorical(Most_frequent, Missing)/
│       ├── Common Techniques for Numerical and Categorical/
│       └── Multivariate Imputation/
├── Handling mixed data/                             # 1 notebook
├── Handling outliers/
│   ├── For Normal Distribution (Z-Score)/
│   ├── For Skewed Distribution (IQR Method)/
│   └── Percentile Method (Setting thresholds)/
├── Mathematical Transformations/                    # 2 notebooks
├── PCA (Principle Component Analysis)/              # 2 notebooks
├── Scikit-Learn Pipelines/
│   ├── Without Using Pipeline/                      # manual transformers + saved .pkl artifacts
│   └── With Pipeline/                               # single end-to-end pipeline.pkl
├── data/                                            # ⬅ you create this (see Dataset Setup; git-ignored)
└── .gitignore
```

---

## 📚 Module Breakdown

### 🧩 1. Column Transformer
[`Column Transformer/`](./Column%20Transformer)
- Builds the same preprocessing three ways: **manually**, with **`ColumnTransformer`**, and with a **`Pipeline` nested inside a `ColumnTransformer`**.
- Combines `SimpleImputer`, `OrdinalEncoder` (explicit category order), `OneHotEncoder(drop='first')`, and `StandardScaler` on mixed-type columns.

### 🔢 2. Encoding Numerical Features
[`Encoding numerical features/`](./Encoding%20numerical%20features)
- **Discretization (binning)** with `KBinsDiscretizer` (quantile strategy, ordinal encoding).
- **Binarization** with `Binarizer` for threshold-based flags.
- Evaluates impact with `DecisionTreeClassifier` and cross-validated accuracy.

### 🏗️ 3. Feature Construction & Splitting
[`Feature Construction and Feature Splitting/`](./Feature%20Construction%20and%20Feature%20Splitting)
- **Construction:** derive new signals (e.g., family size and family type from `SibSp` + `Parch`).
- **Splitting:** extract structured parts from compound strings (e.g., passenger titles from names).
- Compares model performance **before vs. after** engineering with cross-validated `LogisticRegression`.

### 📏 4. Scaling & Normalization
[`Feature Scaling - Normalization/`](./Feature%20Scaling%20-%20Normalization) · [`Feature Scaling - Standardization/`](./Feature%20Scaling%20-%20Standardization)
- **Min-Max scaling** (`MinMaxScaler`) and **Standardization / z-score** (`StandardScaler`).
- Before/after **KDE distribution plots** to show what scaling does (and doesn't) change.

### 🏷️ 5. Handling Categorical Variables
[`Handling Categorical Variables/`](./Handling%20Categorical%20Variables)
- **Label & Ordinal Encoding** with explicit, meaningful category ordering.
- **One-Hot Encoding** via `pd.get_dummies` *and* `sklearn.OneHotEncoder`, `drop='first'` to avoid the dummy-variable trap, and a strategy for **high-cardinality features** (grouping rare categories via a frequency threshold).

### 🕒 6. Date & Time Variables
[`Handling Date and Time variables/`](./Handling%20Date%20and%20Time%20variables)
- Parses timestamps with `pd.to_datetime` and the `.dt` accessor.
- Extracts **year, month (number/name), day-of-month, day-of-week (number/name), quarter, semester, weekend flags**, and more from real-world messages/orders data.

### 🕳️ 7. Missing Data & Imputation
[`Handling Missing Data/`](./Handling%20Missing%20Data)

| Sub-topic | Methods covered |
|---|---|
| **Complete Case Analysis** | Listwise deletion and its effect on feature distributions |
| **Numerical imputation** | Mean / median imputation, arbitrary-value imputation (`SimpleImputer` + `ColumnTransformer`) |
| **Categorical imputation** | Most-frequent (mode) imputation, "Missing" category imputation — pandas **and** sklearn implementations, with before/after distribution checks |
| **Common techniques** | Random-sample imputation (numerical & categorical), `MissingIndicator` / `add_indicator=True`, and **`GridSearchCV`** to select the best imputation strategy |
| **Multivariate imputation** | `KNNImputer` and `IterativeImputer` (MICE-style) — including a **from-scratch** implementation beside the sklearn one |

### 🧬 8. Handling Mixed Data
[`Handling mixed data/`](./Handling%20mixed%20data)
- Two real-world patterns: **numeric + categorical in the same column** (separated with `pd.to_numeric(errors='coerce')` + `np.where`) and **alphanumeric codes** (e.g., `Cabin`, `Ticket`) split into numeric and categorical parts.

### 🚨 9. Outlier Detection & Treatment
[`Handling outliers/`](./Handling%20outliers)
- **Z-Score method** (mean ± 3σ) for roughly normal data.
- **IQR / proximity rule** for skewed data.
- **Percentile method** (1st/99th) with **winsorization**.
- Each method demonstrates both **trimming** (removal) and **capping** (clipping to boundary values).

### 📐 10. Mathematical Transformations
[`Mathematical Transformations/`](./Mathematical%20Transformations)
- **Function transforms** with `FunctionTransformer` (e.g., `log1p`) inside a `ColumnTransformer`.
- **Power transforms** with `PowerTransformer` — **Box-Cox** and **Yeo-Johnson**.
- Quantifies gains with accuracy / R² under cross-validation.

### 🔻 11. PCA — Principal Component Analysis
[`PCA (Principle Component Analysis)/`](./PCA%20%28Principle%20Component%20Analysis%29)
- **Step-by-step PCA from scratch:** mean-centering → covariance matrix → eigen-decomposition (`np.linalg.eigh`) → projection, with 3D visualization.
- **PCA on MNIST** with `sklearn.decomposition.PCA` (`n_components` selection, 2D/3D projections) and a `KNeighborsClassifier` to measure accuracy vs. dimensionality.

### 🏭 12. Scikit-Learn Pipelines
[`Scikit-Learn Pipelines/`](./Scikit-Learn%20Pipelines)
- **Without Pipeline:** manually fitted imputers/encoders/scalers, each persisted as a separate `.pkl` — shows how fragile and error-prone this is at inference time.
- **With Pipeline:** one `Pipeline` + `ColumnTransformer` (imputation → one-hot → min-max scaling → `SelectKBest(chi2)` → `DecisionTreeClassifier`) saved as a **single artifact** and reloaded in a separate testing notebook for inference.

---

## 💾 Dataset Setup Guide

To keep the repository **lightweight and fast to clone**, datasets are **not stored in Git** — they are hosted on Google Drive, and the `data/` folder is git-ignored.

📥 **Download:** [Google Drive — Datasets](https://drive.google.com/drive/folders/1GO2G70qKMiCWuPX5GijOWM9Q8jKN2GDg)

### Option A — Manual download (browser)

1. Open the Google Drive link above.
2. Click the folder name (top bar) → **Download**. Google Drive bundles everything into a `.zip` archive.
3. Extract the archive.
4. Create a folder named **`data`** in the **root of the cloned repository** and place all dataset files **directly inside it**.

> ⚠️ The files must sit at `data/<file>.csv`, **not** `data/data/<file>.csv`. If extraction created a nested folder, move the files up one level.

### Option B — Command line (`gdown`)

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1GO2G70qKMiCWuPX5GijOWM9Q8jKN2GDg" -O data
```

### ✅ Expected layout

```text
Feature-Engineering/
├── data/
│   ├── 50_Startups.csv
│   ├── Titanic_dataset.csv
│   ├── cars.csv
│   ├── concrete_data.csv
│   ├── covid_toy.csv
│   ├── customer.csv
│   ├── data_science_job.csv
│   ├── messages.csv
│   ├── orders.csv
│   ├── placement.csv
│   ├── titanic.csv
│   ├── titanic_toy.csv
│   ├── train.csv              # House Prices (Ames) dataset
│   ├── weight-height.csv
│   └── wine_data.csv
├── Column Transformer/
└── ...
```

Notebooks reference this folder with relative paths (e.g., `../data/...`, `../../data/...`, `../../../data/...`, depending on folder depth), so **no path edits are needed** once `data/` is in the repo root.

### 🔍 Quick sanity check

Run from the repository root:

```python
from pathlib import Path

expected = [
    "50_Startups.csv", "Titanic_dataset.csv", "cars.csv", "concrete_data.csv",
    "covid_toy.csv", "customer.csv", "data_science_job.csv", "messages.csv",
    "orders.csv", "placement.csv", "titanic.csv", "titanic_toy.csv",
    "train.csv", "weight-height.csv", "wine_data.csv",
]
missing = [f for f in expected if not (Path("data") / f).exists()]
print("✅ All datasets found!" if not missing else f"❌ Missing: {missing}")
```

### 🖼️ Note on the MNIST notebook

`PCA (Principle Component Analysis)/02_pca-on-mnist-dataset.ipynb` was authored on **Kaggle** and reads the *Digit Recognizer* `train.csv` from Kaggle's input directory. Either run it on Kaggle (link inside the notebook), or download the [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) data and point `pd.read_csv(...)` to your local copy.

---

## ⚙️ Installation & Execution

### 1️⃣ Clone the repository

```bash
git clone https://github.com/AryanGoyal17/Feature-Engineering.git
cd Feature-Engineering
```

### 2️⃣ Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3️⃣ Install dependencies

Create a `requirements.txt` in the repo root:

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
seaborn>=0.12
matplotlib>=3.7
scipy>=1.10
plotly>=5.0
notebook>=7.0
ipykernel>=6.0
```

Then install:

```bash
pip install -r requirements.txt
```

Or in a single command:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib scipy plotly notebook ipykernel
```

### 4️⃣ Add the datasets

Follow the [Dataset Setup Guide](#-dataset-setup-guide) so that `data/` exists in the repo root.

### 5️⃣ Launch

**Jupyter Notebook**
```bash
jupyter notebook
```

**VS Code** — open the folder, select the `.venv` interpreter as the notebook kernel, and run any `.ipynb` file.

> ⚠️ **Pickle compatibility:** the `.pkl` files in `Scikit-Learn Pipelines/` were serialized with a specific scikit-learn version. If loading fails with a version warning/error, simply re-run the corresponding `01_preprocessing_training.ipynb` to regenerate the artifacts in your environment.

---

## 🎯 Key Methodological Highlights

- 🛡️ **Leakage-aware workflow** — in the scaling, encoding, imputation, `ColumnTransformer`, and pipeline modules, data is split first; transformers are **fitted on `X_train` only** and merely applied to `X_test`.
- 🎲 **Deterministic & reproducible** — `random_state=42` is used for train/test splits and for random-sample imputation, so results are identical run-to-run (a prerequisite for consistent behavior in production).
- 🔁 **Two implementations, side by side** — pandas vs. scikit-learn (`get_dummies` vs. `OneHotEncoder`, manual vs. `SimpleImputer`), and **from-scratch vs. library** for `IterativeImputer` and PCA, to build real understanding rather than API familiarity.
- 📊 **Evidence over intuition** — each technique is checked with distribution plots (KDE/histograms), covariance comparisons after imputation, and downstream metrics (accuracy, R²) via `cross_val_score`.
- 🔧 **Hyperparameter-tuned preprocessing** — `GridSearchCV` is used to *choose* imputation strategies, treating preprocessing as part of the model.
- 🏭 **Pipeline integration** — `Pipeline` + `ColumnTransformer` (including pipelines nested in transformers) produce a **single serializable artifact**, contrasted against the fragile multi-file "no pipeline" approach.
- 🧱 **Robust encoding practices** — `handle_unknown='ignore'` for unseen categories, `drop='first'` to avoid multicollinearity, explicit category order for ordinal features, and rare-category grouping for high cardinality.
- 🧼 **Modern Pandas / Seaborn idioms** — `.dt` accessor for datetimes, `pd.to_numeric(errors='coerce')`, vectorized `np.where`, and `sns.histplot` / `sns.kdeplot` (in place of the deprecated `distplot`).

---

## 🧭 Suggested Learning Path

| Stage | Modules |
|---|---|
| **1. Clean** | Missing Data → Mixed Data → Outliers |
| **2. Encode** | Categorical Variables → Encoding Numerical Features → Date & Time |
| **3. Transform** | Scaling → Mathematical Transformations → Feature Construction & Splitting |
| **4. Reduce** | PCA |
| **5. Productionize** | Column Transformer → Scikit-Learn Pipelines |

---

## 👤 Author & Connect

**Aryan Goyal**
*B.E. Artificial Intelligence & Data Science · Machine Learning Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-AryanGoyal17-181717?logo=github&logoColor=white)](https://github.com/AryanGoyal17)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aryan-goyal17/)

---

<div align="center">

⭐ **If this repository helped you, consider giving it a star!** ⭐

Found an issue or have a suggestion? Feel free to open an [issue](https://github.com/AryanGoyal17/Feature-Engineering/issues) or submit a pull request.

</div>