# Cardiac Patient Monitoring System

## Overview

The **Cardiac Patient Monitoring System** is an applied machine learning project developed as part of the BinX Tech AI & ML Internship program. The project implements a complete, curriculum-aligned supervised binary classification workflow to predict heart disease risk using clinical, demographic, and electrocardiogram stress-testing data.

The project implements an end-to-end machine learning workflow, encompassing environment setup, data auditing and cleaning, comprehensive exploratory data analysis (EDA), domain-informed feature engineering, baseline and comparison classification modeling, hyperparameter optimization with `RandomizedSearchCV`, stratified cross-validation, and leak-free Scikit-learn preprocessing pipelines.

---

## Project Objectives

1. **Environment & Data Inspection:** Set up a reproducible Python environment, audit dataset schema and data types, and document feature characteristics.
2. **Data Preparation:** Identify and treat data anomalies, structural missing values (172 unrecorded zeros in `Cholesterol`, 1 zero in `RestingBP`), and categorical features.
3. **Exploratory Data Analysis & Statistics:** Compute descriptive statistics, visualize univariate and bivariate distributions, apply the Interquartile Range (IQR) method for outlier screening, evaluate correlation heatmaps, and document clinical data storytelling.
4. **Supervised Classification & Baseline:** Formulate a binary classification problem predicting `HeartDisease` (1 = Heart Disease, 0 = Normal) and establish baseline performance using Logistic Regression on a stratified 80/20 train/test split.
5. **Feature Engineering & Hyperparameter Tuning:** Engineer physiological ratios, clinical risk interactions, and pathology flags, and optimize `RandomForestClassifier` hyperparameters using `RandomizedSearchCV` with 5-fold stratified cross-validation.
6. **Leak-Free Scikit-learn Pipeline:** Encapsulate data imputation, standardization, one-hot encoding, and classification into a unified `Pipeline` via `ColumnTransformer` to prevent test-set data leakage.
7. **Clinical Model Evaluation:** Evaluate models across Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC, explaining False Positives and False Negatives in plain medical terminology.

---

## Dataset Description

- **Dataset File:** [`Data/heart.csv`](file:///c:/Users/Amr-Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Data/heart.csv)
- **Total Records:** 918 patient observations
- **Total Features:** 11 clinical and physiological attributes + 1 binary target variable
- **Target Variable (`HeartDisease`):**
  - `0` (Normal / No Heart Disease): 410 records (~44.66%)
  - `1` (Heart Disease Present): 508 records (~55.34%)

### Data Dictionary

| Column Name | Data Type | Description | Values / Range | Missing Count |
| :--- | :--- | :--- | :--- | :--- |
| `Age` | Integer / Continuous | Patient age in years | 28 to 77 | 0 (0.0%) |
| `Sex` | Categorical | Biological sex | `M` (Male), `F` (Female) | 0 (0.0%) |
| `ChestPainType` | Categorical | Clinical chest pain presentation | `ASY` (Asymptomatic), `NAP` (Non-Anginal), `ATA` (Atypical), `TA` (Typical) | 0 (0.0%) |
| `RestingBP` | Integer / Continuous | Resting blood pressure (mm Hg) | 0 to 200 (1 zero anomaly imputed) | 0 (0.0%) |
| `Cholesterol` | Integer / Continuous | Serum cholesterol level (mm/dl) | 0 to 603 (172 zero values imputed) | 0 (0.0%) |
| `FastingBS` | Binary | Fasting blood sugar threshold | `1` if FastingBS > 120 mg/dl, `0` otherwise | 0 (0.0%) |
| `RestingECG` | Categorical | Resting electrocardiogram findings | `Normal`, `ST` (ST-T abnormality), `LVH` (Left Ventricular Hypertrophy) | 0 (0.0%) |
| `MaxHR` | Integer / Continuous | Maximum heart rate achieved during exercise (bpm) | 60 to 202 | 0 (0.0%) |
| `ExerciseAngina` | Categorical | Exercise-induced angina | `Y` (Yes), `N` (No) | 0 (0.0%) |
| `Oldpeak` | Float / Continuous | Exercise-induced ST depression relative to rest | -2.6 to 6.2 | 0 (0.0%) |
| `ST_Slope` | Categorical | Peak exercise ST segment slope | `Up` (Upsloping), `Flat` (Flat), `Down` (Downsloping) | 0 (0.0%) |
| `HeartDisease` | Binary (Target) | Diagnostic classification label | `0` = Normal, `1` = Heart Disease | 0 (0.0%) |

---

## Key Exploratory Data Analysis (EDA) Insights

1. **Target Balance:** The dataset is well-balanced (55.3% disease vs. 44.7% normal), ensuring balanced cross-validation without extreme class skew.
2. **Asymptomatic Risk:** Patients with `ChestPainType = ASY` represent the highest cardiac disease concentration (79.0%), indicating silent myocardial ischemia.
3. **ST Segment Slope:** An `Up` ST slope strongly associates with healthy status (80.3% normal), whereas `Flat` and `Down` slopes indicate acute cardiac risk (82.8% and 77.8% disease rate).
4. **Stress Testing Markers:** Lower maximum heart rate (`MaxHR`, $r = -0.40$) and higher ST depression (`Oldpeak`, $r = +0.40$) correlate moderately with heart disease.
5. **Data Quality Treatment:** 172 records with 0 mg/dl cholesterol and 1 record with 0 mm Hg blood pressure were treated as structural missingness and imputed with median values during preprocessing.

---

## Domain Feature Engineering

To enhance model predictive power, domain-informed features were engineered:

1. **`MaxHR_Ratio`:** $\frac{\text{MaxHR}}{220 - \text{Age}}$ (quantifies exercise capacity and chronotropic reserve).
2. **`BP_Age_Ratio`:** $\frac{\text{RestingBP}}{\text{Age}}$ (normalizes blood pressure against biological age).
3. **`Exercise_Stress_Risk`:** Binary flag for `ExerciseAngina == 'Y'` AND `ST_Slope.isin(['Flat', 'Down'])`.
4. **`Metabolic_Risk`:** Binary flag for `FastingBS == 1` AND `RestingBP >= 140`.
5. **`Is_Oldpeak_Significant`:** Binary indicator for ischemic ST depression ($\text{Oldpeak} \ge 1.5\text{ mm}$).
6. **`Is_Asymptomatic`:** Indicator for high-risk asymptomatic presentation (`ChestPainType == 'ASY'`).

---

## Model Evaluation & Performance Comparison

Models were trained on an 80% training split ($n = 734$) and evaluated on a held-out 20% test split ($n = 184$) with 5-fold stratified cross-validation (`StratifiedKFold`).

| Model Configuration | 5-Fold CV F1-Score | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Logistic Regression** | 0.8541 | 88.59% | 89.00% | 91.00% | 90.00% | 0.9152 |
| **Tuned Random Forest (`RandomizedSearchCV`)** | **0.8751** | **85.33%** | **88.00%** | **85.00%** | **86.57%** | **0.9163** |
| **End-to-End Scikit-learn Pipeline** | 0.8620 | 85.87% | 87.38% | 88.24% | 87.38% | 0.9301 |

### Best Random Forest Hyperparameters:
- `n_estimators`: 100
- `max_depth`: 3
- `max_features`: `log2`
- `min_samples_split`: 5
- `min_samples_leaf`: 4
- `criterion`: `gini`

---

## Plain-Language Clinical Error Interpretation

In clinical cardiac monitoring, classification errors carry asymmetric medical consequences:

1. **False Positives (Type I Error):**
   - **Definition:** A healthy patient is predicted to have heart disease.
   - **Clinical Consequence:** Results in psychological anxiety, further diagnostic testing (e.g., echocardiogram, coronary computed tomography), and extra financial cost. It does not pose direct physical harm or mortality risk.

2. **False Negatives (Type II Error):**
   - **Definition:** A cardiac patient with heart disease is mistakenly predicted as normal.
   - **Clinical Consequence:** Highly dangerous. The patient is discharged without essential medical therapy (statins, antiplatelet agents, lifestyle intervention), creating severe risk of an acute, unmonitored cardiovascular event.
   - **Clinical Priority:** Model tuning emphasizes high **Recall / Sensitivity** to minimize False Negatives.

---

## Project Directory Structure

```text
cardiac_monitoring_project/
├── Data/
│   ├── README.md                              # Dataset documentation and data dictionary
│   └── heart.csv                              # Clinical dataset (918 records)
├── Notebooks/
│   └── Cardiac_Patient_Monitoring_System.ipynb # End-to-end analysis and modeling notebook
├── Outputs/                                   # High-resolution EDA visualization charts
│   ├── Bivariate Categorical features vs Target.png
│   ├── Bivariate Numerical features vs Target.png
│   ├── Correlation Matrix and Heatmap.png
│   ├── Multivariate Feature Screening (Pairplot).png
│   ├── Univariate Categorical distributions.png
│   └── Univariate Numerical distributions.png
├── requirements.txt                           # Python dependencies
└── README.md                                  # Complete project documentation and report
```

---

## Setup & Execution Guide

### Prerequisites
- Python 3.10 or higher
- Virtual environment tool (`venv`)
- PowerShell / Command Line

### 1. Activate Virtual Environment
From the workspace root directory:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
Launch Jupyter Notebook and run all cells sequentially from top to bottom:
```powershell
jupyter notebook Notebooks/Cardiac_Patient_Monitoring_System.ipynb
```