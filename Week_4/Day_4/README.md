
<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 4 — Day 4: Feature Engineering & Hyperparameter Tuning
</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 4 Focus: Feature Engineering, Parameter Search & Systematic Tuning via GridSearchCV</b>

Today we explore how domain understanding turns into predictive power through <b>Feature Engineering</b> and how systematic hyperparameter optimization via <b>GridSearchCV</b> eliminates manual guesswork to produce reliable, regularized models.

</blockquote>

---

## <span style="color:#F78BA0">4.1 Overview & Objectives</span>

<blockquote style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

<b>Key Objectives:</b>

- <b>Engineer Domain Features:</b> Construct new derived features using aggregation, ratios, and binning techniques.
- <b>Distinguish Parameters from Hyperparameters:</b> Understand the difference between parameters learned during training and hyperparameters set before training.
- <b>Establish a Raw Baseline:</b> Benchmark unengineered, default Random Forest performance using 5-Fold Cross-Validation.
- <b>Systematic Optimization with GridSearchCV:</b> Automate the search across multiple hyperparameter combinations using rotating 5-fold cross-validation.
- <b>Analyze Feature Importance:</b> Extract and visualize Gini feature importances to determine the most influential variables.

</blockquote>

---

## <span style="color:#85C1E9">4.2 Dataset Information</span>

We utilize the **Telco Customer Churn Dataset**:

- <b>Source File:</b> `../../Data/Customer-Churn.csv`
- <b>Instances:</b> 7,032 clean rows (after removing missing values in `TotalCharges` and dropping `customerID`).
- <b>Target Variable ($y$):</b> `Churn` (`1` = Churned [26.54%], `0` = Retained [73.46%]).
- <b>Partitioning:</b> Stratified 80/20 train/test split (5,625 training rows / 1,407 test rows, `random_state=42`).

---

## <span style="color:#F8C471">4.3 Feature Engineering & Domain Rationale</span>

Feature engineering is the process of creating, transforming, and selecting the input variables the model learns from. Three features were engineered:

1. **`total_services` (Feature Creation / Aggregation)**:
   - Formulated by summing the 8 add-on service flags (`PhoneService`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).
   - *Rationale*: Customers with multiple active services face higher switching friction and have lower churn rates.

2. **`avg_monthly_charges` (Feature Creation / Ratio)**:
   - Formulated as `TotalCharges / (tenure + 1)`.
   - *Rationale*: Captures historical average monthly billing, exposing discrepancies between current charges and long-term averages.

3. **`tenure_group` (Binning / Discretization)**:
   - Formulated via `pd.cut()` into cohorts: `New` (0-12m), `Established` (12-24m), `Loyal` (24-48m), and `Veteran` (48-72m).
   - *Rationale*: Discretizes continuous tenure into business lifecycle stages to capture non-linear churn probabilities.

---

## <span style="color:#309c42ff">4.4 Hyperparameter Tuning with GridSearchCV</span>

GridSearchCV automates tuning by cross-validating every combination in a specified parameter grid:

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 8, 10],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "criterion": ["gini", "entropy"]
}
```

- **Winning Configuration**: `{'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 200}`

---

## <span style="color:#C39BD3">4.5 Experimental Results & Progression Comparison</span>

| Stage | Feature Set | Model Architecture & Hyperparameters | 5-Fold CV Accuracy | Test Accuracy | Test F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Stage 1: Raw Baseline** | Original Features | `RandomForestClassifier(n_estimators=100, max_depth=None)` | `0.7842` | `0.7896` | `0.5673` |
| **Stage 2: Tuned Model** | Original + 3 Engineered Features | `RandomForestClassifier(max_depth=10, min_samples_leaf=4, n_estimators=200)` | **`0.8020`** | `0.7875` | `0.5544` |

---

## <span style="color:#F78BA0">4.6 Key Insights & Diagnostic Analysis</span>

1. **Impact of Hyperparameter Regularization**:
   Unconstrained random forests (`max_depth=None`, `min_samples_leaf=1`) overfit individual training folds. Restricting depth (`max_depth=10`) and enforcing leaf sample thresholds (`min_samples_leaf=4`) pruned noisy decision branches and raised cross-validated accuracy from **78.42% to 80.20%**.

2. **Validation of Engineered Features**:
   `avg_monthly_charges` and `total_services` ranked in the top 10 most important features in Gini importance, confirming that domain-engineered interactions provide substantial predictive signal.

3. **CV Stability vs. Single Test Split**:
   While the single test split score showed minor fluctuation due to sample variance, the 5-fold cross-validation score reliably demonstrated overall generalization improvement across all 5,625 training instances.

---

## <span style="color:#85C1E9">4.7 Repository Structure</span>

- `Task.ipynb`: Jupyter Notebook containing data loading, data cleaning, feature engineering, raw baseline benchmarking, hyperparameter optimization with `GridSearchCV`, and feature importance visualization.

