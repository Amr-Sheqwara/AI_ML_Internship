<h1  style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">

AI & ML Course with BinX — Week 4 — Day 5: Scikit-Learn Pipelines & Tuned Mini-Project

</h1>

<blockquote  style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 5 Focus: Leak-Free Workflows, ColumnTransformer, and End-to-End Tuning via GridSearchCV</b>

Today marks the culmination of Week 4 with the Tuned Pipeline Mini-Project. We assemble an end-to-end, leak-free machine learning architecture integrating domain-informed feature engineering, heterogeneous feature transformation via <b>ColumnTransformer</b>, hyperparameter optimization across the pipeline using <b>GridSearchCV</b>, and a final evaluation on an untouched held-out test set.

</blockquote>

---

## <span style="color:#F78BA0">5.1 Overview & Objectives</span>

<blockquote  style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

<b>Key Objectives:</b>

-  <b>Prevent Data Leakage:</b> Understand how manual preprocessing causes information leakage and how chaining steps inside a Scikit-Learn `Pipeline` makes leakage structurally impossible.

-  <b>Handle Heterogeneous Data:</b> Use `ColumnTransformer` to apply distinct transformations (standard scaling for numeric features and one-hot encoding for categorical features) simultaneously.

-  <b>Integrate Domain Features:</b> Incorporate the engineered features developed during Day 4 directly into the pipeline workflow.

-  <b>Tune Full Pipelines with GridSearchCV:</b> Optimize hyperparameters across preprocessing and estimator stages simultaneously using double-underscore parameter syntax (`model__<parameter>`).

-  <b>Execute Held-Out Test Evaluation:</b> Maintain strict separation of evaluation sets by assessing the best tuned pipeline once on the untouched held-out test set.

</blockquote>

---

## <span style="color:#85C1E9">5.2 Dataset Information</span>

We utilize the **Telco Customer Churn Dataset**:

-  <b>Source File:</b> `../../Data/Customer-Churn.csv`

-  <b>Instances:</b> 7,032 clean rows (after coercing whitespace in `TotalCharges` to numeric, dropping null values, and dropping `customerID`).

-  <b>Target Variable ($y$):</b> `Churn` (`1` = Churned [26.54%], `0` = Retained [73.46%]).

-  <b>Partitioning:</b> Stratified 80/20 train/test split (5,625 training instances / 1,407 test instances, `random_state=42`).

---


## <span style="color:#F8C471">5.3 Pipeline Architecture & Data Leakage Prevention</span>

### 1. Why Pipelines Exist

When preprocessing (scaling, encoding) and modeling are conducted as separate manual steps, data leakage sneaks into the model. If a scaler is fitted on the whole dataset before splitting, or fitted on validation folds during cross-validation, information from test observations leaks into training.

A `Pipeline` chains preprocessing and modeling into a single estimator:

- When calling `pipeline.fit(X_train, y_train)`, the scaler and encoder are fitted on the training data only.
- During cross-validation, each fold is transformed using only that fold's training portion.

- When calling `pipeline.predict(X_test)`, the test data is transformed using the parameters learned from the training set, eliminating leakage.

### 2. ColumnTransformer for Mixed Data Types

Real-world datasets contain mixed numeric and categorical variables requiring different preprocessing:

-  **Numeric Features (`6`)**: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, `total_services`, `avg_monthly_charges` are standardized via `StandardScaler`.

-  **Categorical Features (`16`)**: Demographic, service, contract attributes, and binned `tenure_group` are encoded via `OneHotEncoder(handle_unknown='ignore')`.

```python
preprocessor = ColumnTransformer(
transformers=[
("num", StandardScaler(), numeric_cols),
("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])


pipeline = Pipeline(
steps=[
("pre", preprocessor),
("model", RandomForestClassifier(random_state=42))
])
```

---

## <span style="color:#309c42ff">5.4 Hyperparameter Optimization with GridSearchCV</span>

`GridSearchCV` searches over hyperparameter combinations across the pipeline using 5-fold stratified cross-validation. Pipeline step parameters are addressed using the double-underscore prefix:

```python
param_grid = {
"model__n_estimators": [100, 200],
"model__max_depth": [5, 10, 15],
"model__min_samples_leaf": [2, 4],
"model__criterion": ["gini", "entropy"]
}
  
grid = GridSearchCV(
estimator=pipeline,
param_grid=param_grid,
cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
scoring="f1",
n_jobs=-1,
return_train_score=True
)

grid.fit(X_train, y_train)
```

---
  
## <span style="color:#C39BD3">5.5 Experimental Results & Progression Comparison</span>

| Evaluation Stage | Workflow Architecture | Configuration | Cross-Validated Score (5-Fold CV) | Test Set Accuracy | Test Set F1-Score | Test Set ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Stage 1: Raw Baseline** | ColumnTransformer + Pipeline | `RandomForestClassifier(default)` | `F1: 0.5601 +/- 0.0220` (Acc: `0.7929`) | N/A | N/A | N/A |
| **Stage 2: Tuned Pipeline** | ColumnTransformer + Pipeline + GridSearchCV | `max_depth=15, min_samples_leaf=2, n_estimators=200` | **`F1: 0.5832`** | **`0.7910`** | **`0.5651`** | **`0.8267`** |

---

## <span style="color:#F78BA0">5.6 Key Insights & Best Practices</span>

1.  **Structural Elimination of Leakage**:
  
Encapsulating `StandardScaler` and `OneHotEncoder` within `ColumnTransformer` inside `Pipeline` guarantees that feature scaling statistics (mean, standard deviation) and categorical encoding mappings are computed strictly from the training folds, preserving the absolute integrity of validation and test folds.

2.  **Cross-Validated Hyperparameter Selection**:

Evaluating parameter combinations across rotating 5-fold stratified splits protected against selecting hyperparameter values based on a single lucky split, improving the cross-validated F1-score from **0.5601 to 0.5832**.

3.  **Generalization on Held-Out Data**:

Evaluating the winning pipeline estimator (`grid.best_estimator_`) once on the untouched held-out test set yielded consistent performance (**Accuracy: 79.10%**, **F1-Score: 0.5651**, **ROC-AUC: 0.8267**), validating that the model generalizes effectively without overfitting.

---

## <span style="color:#85C1E9">5.7 Repository Structure</span>

-  `Task.ipynb`: Jupyter Notebook containing the full workflow: data loading, data cleaning, feature engineering, pipeline creation with `ColumnTransformer`, baseline 5-fold cross-validation, hyperparameter optimization with `GridSearchCV`, held-out test set evaluation, and confusion matrix visualization.