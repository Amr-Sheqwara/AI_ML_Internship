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
8. **Unsupervised Analysis & Latent Phenotyping:** Apply K-Means clustering, DBSCAN density-based clustering, Principal Component Analysis (PCA), and t-Distributed Stochastic Neighbor Embedding (t-SNE) to discover patient risk sub-phenotypes, isolate clinical anomalies, reduce feature dimensionality, and visualize non-linear patient manifolds without diagnostic labels.

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

## Unsupervised Learning: Clustering, Anomaly Detection, Dimensionality Reduction & Manifold Learning (Milestone 6)

In addition to supervised classification, unsupervised learning was applied to uncover latent patient sub-phenotypes, isolate clinical anomalies, reduce feature dimensionality, and visualize non-linear patient manifolds without using target labels (`HeartDisease`).

### 1. Feature Standardization
Continuous physiological features (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) were standardized using `StandardScaler` ($\mu = 0, \sigma = 1$) to eliminate scale dominance in Euclidean distance calculations.

### 2. Centroid-Based Clustering (K-Means & Patient Risk Phenotyping)
- **The Elbow Method & Silhouette Validation:** Evaluated across $k \in [1, 10]$ with `random_state=42` and `n_init=10`. Quantitative Silhouette analysis demonstrates that **$k=2$ achieves the global maximum Silhouette Score ($0.2150$)**, naturally bifurcating the patient cohort into two distinct cardiovascular risk phenotypes.
- **Discovered Patient Risk Subgroups ($k=2$):**

| Cluster | Key Clinical Characteristics | Cohort Size | Disease Rate | Cardiovascular Risk Profile |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | Older age (mean 59.5 yrs), elevated blood pressure (`RestingBP` 139.6 mmHg), reduced exercise capacity (`MaxHR` 122.4 bpm), severe ST depression (`Oldpeak` 1.38 mm) | 466 (50.76%) | **76.39%** | **High Cardiovascular Risk & Exercise Ischemia Subgroup** |
| **Cluster 1** | Younger age (mean 47.3 yrs), controlled blood pressure (`RestingBP` 124.9 mmHg), robust exercise capacity (`MaxHR` 151.7 bpm), minimal ST depression (`Oldpeak` 0.38 mm) | 452 (49.24%) | **33.63%** | **Low Cardiovascular Risk / Preserved Reserve Subgroup** |

---

### 3. Density-Based Clustering & Outlier Isolation (DBSCAN)
- **Hyperparameter Calibration:** Evaluated sorted 5th-nearest-neighbor distances (k-distance graph) to select $\varepsilon = 1.3$ with $\text{MinPts} = 5$.
- **Clinical Anomaly Isolation ($label = -1$):** DBSCAN automatically flagged **51 patient records (5.56% of cohort)** as density noise/outliers without ground-truth labels.
- **Pathological Profiling of Flagged Outliers:**
  - **Disease Concentration:** The **Heart Disease prevalence in the DBSCAN outlier cohort is 78.43%** (vs. 53.98% in core patients).
  - **Extreme Clinical Severity:** Outliers capture extreme ischemic ST depression (mean `Oldpeak` 1.95 mm, max 6.2 mm), hypertensive crises (`RestingBP` up to 200 mmHg), and severe hypercholesterolemia (`Cholesterol` up to 603 mg/dl).

---

### 4. Agglomerative Hierarchical Clustering & Dendrogram Analysis
- **Linkage Matrix & Dendrogram Tree:** Computed using Ward's minimum variance criterion on standardized physiological features. Visualized truncated multi-scale merges with a cutoff distance at $d = 33.0$ ($k=2$).
- **Benchmarking vs. K-Means:** Achieves a Silhouette Score of **$0.1465$** (vs. $0.2150$ for K-Means). While hierarchical clustering provides an intuitive multi-level visual taxonomy, K-Means is retained for clinical deployment due to its explicit centroids for real-time patient assignment.

---

### 5. Linear Dimensionality Reduction & Variance Analysis (PCA)
- **Variance Retention Analysis:**
  - **PC1 (34.01% Variance):** Cardiovascular Aging & Exercise Ischemia Axis (High positive loadings on `Age` $+0.601$, `Oldpeak` $+0.450$, `RestingBP` $+0.428$; negative loading on `MaxHR` $-0.494$).
  - **PC2 (21.02% Variance):** Metabolic & Hemodynamic Stress Axis (Dominated by `Cholesterol` $+0.838$, `MaxHR` $+0.394$, `RestingBP` $+0.331$).
  - **Information Retention:** PC1 + PC2 capture **55.03%** of variance; PC1 through PC4 retain **88.47%** of total clinical variance.
- **Downstream Utility:** Orthogonally transforms features to eliminate multicollinearity before linear classification.

---

### 6. Non-Linear Manifold Learning (t-SNE)
- **Local Neighborhood Preservation:** Implemented `TSNE(n_components=2, perplexity=30, random_state=42)` using Student-t kernel mapping to resolve the crowding problem.
- **Clinical Manifold Findings:**
  - Isolates cohesive, dense clusters of high-risk asymptomatic patients (`ChestPainType = ASY`).
  - Confirms non-linear topological separation between healthy reserve patients (Cluster 0) and advanced ischemic patients (Cluster 2).
  - Strictly utilized for exploratory visualization (coordinates lack physical units and are not used for downstream model feature engineering).

---

### 7. Comprehensive Unsupervised Method Comparison Matrix

| Dimension | K-Means Clustering | DBSCAN | Agglomerative Hierarchical | Principal Component Analysis (PCA) | t-SNE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Method Paradigm** | Centroid-Based Partitioning | Density-Based Clustering | Bottom-Up Agglomerative Tree | Linear Orthogonal Projection | Non-Linear Probabilistic Manifold |
| **Mathematical Objective** | Minimizes Within-Cluster Inertia | Connects $\varepsilon$-dense neighborhoods | Minimizes Ward Variance ($\Delta \text{ESS}$) | Maximizes Global Variance ($\mathbf{\Sigma}\mathbf{v} = \lambda\mathbf{v}$) | Minimizes KL Divergence between $P$ and $Q$ |
| **Cluster Geometry** | Spherical, Convex | Arbitrary Shapes & Variable Densities | Hierarchical Nested Partitions | Orthogonal Subspaces | Non-Linear Topological Manifolds |
| **Outlier Handling** | Forced into nearest centroid | Flagged as Noise ($label = -1$) | Merged into small branches | Exerts high leverage on variance | Positioned at manifold periphery |
| **Feature Interpretability** | High (Original clinical units) | High (Original clinical units) | Moderate (Tree branch heights) | Moderate (Linear feature combinations) | Low (Abstract 2D coordinates) |
| **Downstream Utility** | Risk phenotyping & stratification | Anomaly & pathological screening | Multi-scale exploratory review | Multicollinearity removal & compression | Cohort visual validation |
| **Key Cardiac Finding** | Discovers 2 distinct risk cohorts ($33.63\% \to 76.39\%$ disease) | Flags 51 outliers ($78.43\%$ disease rate) | Multi-level tree ($k=2$, sil $0.1465$) | PC1: Aging/Ischemia; PC2: Lipids | Isolates asymptomatic ischemia clusters |

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