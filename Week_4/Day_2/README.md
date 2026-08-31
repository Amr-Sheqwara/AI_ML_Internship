
<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 4 — Day 2: Cross-Validation & Stratification
</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 2 Focus: k-Fold Cross-Validation, Stratification & Stability Diagnostics</b>

Today we transitioned from a single train/validation split to <b>5-Fold Cross-Validation</b>. By using rotating folds and Scikit-Learn <b>Pipelines</b>, we obtained a leak-free, variance-aware estimate of model performance.

</blockquote>

---

## <span style="color:#F78BA0">2.1 Overview & Objectives</span>

-  **$k$-Fold Cross-Validation:** Replace single validation splits with 5 rotating folds.
-  **Stability Metrics:** Evaluate both the **mean** and **standard deviation** across folds.
-  **Stratified $k$-Fold:** Ensure equal class proportions in each validation fold.
-  **Leak-Free Pipeline:** Chain `StandardScaler` and `KNeighborsClassifier(n_neighbors=1)` into a unified `Pipeline`.
-  **Day 1 Comparison:** Benchmark 5-fold CV stability against Day 1's single validation split.

---

## <span style="color:#85C1E9">2.2 Dataset Summary</span>

- **Source File:** `../../Data/processed/penguins_cleaned.csv`
- **Total Instances:** 333 clean rows
- **Split:** 80% CV Training Pool ($N = 266$) / 20% Held-out Test Set ($N = 67$)

---

## <span style="color:#F8C471">2.3 Evaluation Results</span>

| Evaluation Method | Estimator | Mean Accuracy | Standard Deviation ($\sigma$) | Takeaway |
| :--- | :--- | :---: | :---: | :--- |
| **Day 1 Single Split** | `Pipeline(Scaler + KNN k=1)` | `0.9851` | `N/A` | Single point estimate |
| **Day 2 5-Fold CV** | `Pipeline(Scaler + KNN k=1)` | **`0.9925`** | **`± 0.0092`** | **Stable, variance-aware estimate** |
| **Untouched Test Set** | `Pipeline(Scaler + KNN k=1)` | **`1.0000`** | `N/A` | Final generalization audit |

---

## <span style="color:#309c42ff">2.4 Key Takeaways</span>

1. **Why Cross-Validation Beats a Single Split:**
   A single split gives only one point estimate. 5-Fold CV evaluates every sample once across 5 folds, and the small standard deviation ($\pm 0.0092$) confirms the model is reliably generalizing rather than benefiting from lucky data partitioning.
2. **Why Stratification Matters:**
   With class proportions of ~43% Adelie, ~36% Gentoo, and ~21% Chinstrap, `StratifiedKFold` ensures minority classes are equally represented across all training and validation folds.
3. **Leak-Free Scaling:**
   Using `Pipeline` ensures `StandardScaler` calculates mean and variance strictly on the training fold during each CV iteration, avoiding test information leakage.
