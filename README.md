# 🩺 Diabetes Risk Predictor

> **A clinical machine learning pipeline that predicts diabetes risk from routine health and lifestyle indicators — built end-to-end from raw CSV to evaluated model.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualisation-4C72B0)](https://seaborn.pydata.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/code/collinslemeke/diabetes-risk-predictor)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Clinical Context](#clinical-context)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [① Data Collection & Exploration](#-data-collection--exploration)
  - [② Missing Data Audit](#-missing-data-audit)
  - [③ Feature Scaling](#-feature-scaling)
  - [④ Categorical Encoding](#-categorical-encoding)
  - [⑤ Train/Test Split](#-traintest-split)
  - [⑥ Model Training](#-model-training)
  - [⑦ Evaluation](#-evaluation)
- [Performance Metrics](#performance-metrics)
- [Interpretation of Results](#interpretation-of-results)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Repository Structure](#repository-structure)
- [Limitations & Future Work](#limitations--future-work)
- [Author](#author)
- [License](#license)

---

## Overview

**Diabetes Risk Predictor** is a supervised machine learning project that predicts whether an individual is diabetic based on routinely measured clinical and lifestyle indicators: age, body mass index (BMI), HbA1c level, blood glucose level, gender, and smoking history.

The project is implemented as a fully reproducible Kaggle notebook that walks through the complete ML lifecycle — data exploration, preprocessing, feature engineering, model training, and evaluation — producing a logistic regression baseline that achieves **95.91% accuracy** on held-out test data.

It is designed to demonstrate three things:

1. **Clean clinical ML workflow** — every preprocessing step is justified and reversible, suitable for healthcare contexts where data provenance matters.
2. **Honest model evaluation** — accuracy alone is misleading on imbalanced clinical data, so precision, recall, and F1 are reported alongside.
3. **Extensibility** — the codebase imports `RandomForestClassifier` and `SVC` ready for comparative benchmarking, with a structure that supports drop-in algorithm swapping.

---

## Clinical Context

Type 2 diabetes is one of the most common chronic conditions globally, and early identification of at-risk individuals enables preventive intervention before irreversible complications develop. Routine clinical measurements such as **HbA1c** (a 3-month blood sugar average) and **fasting blood glucose** are strong diagnostic indicators, while non-clinical factors such as **BMI**, **age**, and **smoking history** are well-established risk modifiers.

This project builds a screening-style classifier that takes these six features and outputs a binary diabetes prediction. The intended use case is **decision support, not diagnosis** — flagging cases for clinical follow-up, not replacing medical judgement.

---

## Key Results

| Metric | Score |
|---|---|
| **Accuracy** | **95.91%** |
| **Precision** | 86.89% |
| **Recall (Sensitivity)** | 61.36% |
| **F1 Score** | 0.7193 |

Trained on **80%** of the dataset, evaluated on a held-out **20%** test split with `random_state=42` for reproducibility.

---

## Dataset

**Source:** [Kaggle — Diabetes Prediction Dataset](https://www.kaggle.com/datasets) (`diabetes_prediction.csv`)

**Target variable:** `diabetes` (binary: 0 = non-diabetic, 1 = diabetic)

**Features (6):**

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Patient age in years |
| `bmi` | Numeric | Body Mass Index (kg/m²) |
| `HbA1c_level` | Numeric | Glycated haemoglobin level (%) |
| `blood_glucose_level` | Numeric | Fasting blood glucose (mg/dL) |
| `gender` | Categorical | Male / Female / Other |
| `smoking_history` | Categorical | Never / Former / Current / No info / Ever / Not current |

The dataset reflects an imbalanced clinical population (the majority of records are non-diabetic), which is why **recall is reported separately from accuracy** — a high-accuracy classifier on imbalanced data can still miss most of the positive class.

---

## Pipeline Walkthrough

### ① Data Collection & Exploration

The dataset is loaded with pandas, with the implicit Kaggle index used as the unique record identifier. Initial exploration covers:

- `df.shape` — total rows and columns
- `df.info()` — column types and null counts
- `df.head()` — sample inspection
- `df['diabetes'].value_counts()` — class balance audit

### ② Missing Data Audit

Two complementary checks are run before any cleaning:

```python
df.isnull().sum()                       # Numeric summary
sns.heatmap(df.isnull(), cbar=False)    # Visual pattern check
```

The heatmap reveals whether missingness is random or structured (e.g. a whole column missing for a subgroup). Rows with any missing values are dropped, and the cleaned shape is verified.

### ③ Feature Scaling

Continuous clinical features are scaled to the **[0, 1] range** using `MinMaxScaler` to ensure no single feature dominates the model due to magnitude differences (HbA1c is on a 0–15 scale; blood glucose is on a 0–300 scale):

```python
scaler = MinMaxScaler()
columns_to_scale = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
```

### ④ Categorical Encoding

Categorical features are converted via **one-hot encoding** with `pd.get_dummies()`, which produces independent binary columns and avoids the artificial ordinal relationships that a label encoder would introduce:

```python
encoded_data = pd.get_dummies(df, columns=['gender', 'smoking_history'])
```

### ⑤ Train/Test Split

The encoded dataset is split into features (`X`) and target (`y`), then partitioned **80/20** with a fixed random seed for reproducibility:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### ⑥ Model Training

The baseline classifier is **Logistic Regression**, chosen for three reasons:

1. **Interpretability** — coefficients map directly onto feature contributions, important for clinical contexts.
2. **Probabilistic output** — predictions can be thresholded for risk-stratified screening.
3. **Speed and robustness** — converges quickly and resists overfitting on tabular data.

```python
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

The notebook also imports `RandomForestClassifier` and `SVC`, structured to support drop-in comparison benchmarks in future iterations.

### ⑦ Evaluation

Four metrics are computed on the held-out test set:

```python
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
```

---

## Performance Metrics

| Metric | Score | What it tells us |
|---|---|---|
| **Accuracy** | 0.9591 | The model correctly classifies roughly 96% of patients overall. |
| **Precision** | 0.8689 | When the model flags a patient as diabetic, it is correct ~87% of the time (low false-positive rate). |
| **Recall** | 0.6136 | The model identifies ~61% of true diabetic cases. The remaining 39% are missed (false negatives). |
| **F1 Score** | 0.7193 | The harmonic mean of precision and recall, giving a single balanced view. |

---

## Interpretation of Results

The headline 95.91% accuracy is impressive but **must be read alongside recall**, because the dataset is imbalanced toward non-diabetic cases. The lower recall (61.36%) means the model is conservative — it tends to under-flag diabetes rather than over-flag.

In a real clinical setting, this trade-off would need to be inverted: **missing a diabetic case is far more costly than a false alarm**, because a flagged patient simply gets a follow-up test, whereas a missed patient may go untreated. The natural next steps are:

- **Class-weighted training** (`class_weight='balanced'`) to penalise false negatives more heavily.
- **Threshold tuning** — instead of the default 0.5 cutoff, lower the decision threshold to favour recall.
- **Ensemble comparison** — Random Forest and SVC (already imported) typically lift recall on tabular clinical data.
- **SMOTE or under-sampling** to rebalance the training distribution.

---

## Tech Stack

| Layer | Tool |
|---|---|
| **Language** | Python 3.10+ |
| **Data manipulation** | pandas, NumPy |
| **Modelling** | scikit-learn (LogisticRegression, RandomForestClassifier, SVC, tree) |
| **Preprocessing** | scikit-learn (MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder), pandas `get_dummies` |
| **Evaluation** | scikit-learn metrics (accuracy, precision, recall, F1) |
| **Visualisation** | seaborn, matplotlib |
| **Environment** | Kaggle Notebooks (CPU) |

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/CollinsLemeke/diabetes-risk-predictor.git
cd diabetes-risk-predictor
pip install -r requirements.txt
```

**`requirements.txt`:**

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
seaborn>=0.13
matplotlib>=3.8
jupyter>=1.0
```

---

## Running the Notebook

### Option 1 — Run on Kaggle (recommended)

The notebook is published on Kaggle with the dataset already attached:

🔗 **[Open in Kaggle](https://www.kaggle.com/code/collinslemeke/diabetes-risk-predictor)**

Just click **"Copy & Edit"** and run all cells.

### Option 2 — Run locally

1. Download the dataset from Kaggle and place it under `data/diabetes_prediction.csv`.
2. Update the path in the notebook from `/kaggle/input/...` to `data/diabetes_prediction.csv`.
3. Launch Jupyter:

```bash
jupyter notebook Diabetes_classification_Machine_Learning.ipynb
```

4. Run all cells sequentially.

---

## Repository Structure

```
diabetes-risk-predictor/
│
├── Diabetes_classification_Machine_Learning.ipynb   # Main notebook
├── data/
│   └── diabetes_prediction.csv                      # Dataset (not committed; see Kaggle link)
├── requirements.txt                                 # Python dependencies
├── README.md                                        # This file
└── LICENSE                                          # MIT
```

---

## Limitations & Future Work

This is a **baseline implementation** intended for portfolio and educational purposes. Production-grade clinical deployment would require substantial additional work:

- **Class imbalance handling** — apply SMOTE, class weights, or threshold optimisation to lift recall.
- **Cross-validation** — replace single hold-out split with stratified k-fold for more reliable metric estimates.
- **Hyperparameter tuning** — `GridSearchCV` or `Optuna` over Logistic Regression, Random Forest, and SVC.
- **Calibration** — Platt scaling or isotonic regression to ensure predicted probabilities are clinically meaningful.
- **Explainability** — SHAP or LIME explanations per prediction to support clinical decision-making.
- **External validation** — test the model on a different dataset (different geography, demographics) to assess generalisability.
- **Deployment** — wrap the trained model in a Streamlit/FastAPI interface for interactive risk assessment.
- **Regulatory considerations** — under UK MHRA / EU MDR, a clinical decision-support tool would require formal validation, post-market surveillance, and adherence to ISO 13485 / IEC 62304.

---

## Author

**Collins Lemeke**


🔗 [LinkedIn](https://www.linkedin.com/in/collins-lemeke-ai-machine-learning) · [GitHub](https://github.com/CollinsLemeke) · [Kaggle](https://www.kaggle.com/collinslemeke) · [HuggingFace](https://huggingface.co/Lemeke)

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for full terms.

---

> *Built as part of an applied machine learning portfolio focused on clinical and healthcare-adjacent AI. Educational use only — not a medical device.*
