<h1  style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 4 — Day 3: Bias-Variance Trade-Off & Regularization
</h1>

<blockquote  style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 3 Focus: Diagnosing Overfitting, Underfitting & Controlling Model Complexity</b>

Today we explore the foundational machine learning trade-off: the <b>Bias-Variance Trade-Off</b>. Using Decision Trees on the Telco Customer Churn dataset, we demonstrate how unconstrained tree complexity induces high variance (overfitting), how severe oversimplification leads to high bias (underfitting), and how applying pre-pruning regularization techniques shrinks the generalization gap to ensure reliable real-world performance.

</blockquote>

---

## <span  style="color:#F78BA0">3.1 Overview & Objectives</span>

<blockquote  style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

<b>Key Objectives:</b>

- <b>Diagnose High Variance (Overfitting):</b> Train an unconstrained Decision Tree (`max_depth=50`) and quantify the resulting performance disparity between training and testing sets.

- <b>Diagnose High Bias (Underfitting):</b> Restrict model capacity to a single decision stump (`max_depth=1`) to observe inadequate pattern capture.

- <b>Apply Pre-Pruning Regularization:</b> Calibrate hyperparameters (`max_depth=3`, `min_samples_leaf=10`) to restrict tree depth and node partitioning.

- <b>Measure the Generalization Gap:</b> Track the numerical difference between Training Accuracy and Testing Accuracy across different complexity levels.

</blockquote>

---

## <span  style="color:#85C1E9">3.2 Dataset Summary</span>

- <b>Source File:</b> `../../Data/Customer-Churn.csv`

- <b>Target Variable ($y$):</b> `Churn` (`Yes` / `No`)

- <b>Cleaning Steps:</b> Coerced whitespace values in `TotalCharges` to numeric and dropped nulls; dropped arbitrary identifier `customerID`; applied one-hot encoding via `pd.get_dummies(drop_first=True)`.

- <b>Partitioning:</b> 70% Training Set / 30% Test Set using `train_test_split` with `random_state=42` and `stratify=y`.

---

## <span  style="color:#F8C471">3.3 Experimental Results & Generalization Gap</span>

| Model Architecture | Hyperparameter Configuration | Train Accuracy | Test Accuracy | Generalization Gap | Diagnostic State |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Unconstrained Tree** | `max_depth=50` | `0.9988` | `0.7047` | `0.2941 (29.41%)` | **Overfitting (High Variance)** |
| **Decision Stump** | `max_depth=1` | `0.7343` | `0.7341` | `0.0002 (0.02%)` | **Underfitting (High Bias)** |
| **Regularized Tree** | `max_depth=3, min_samples_leaf=10` | **`0.7877`** | **`0.7791`** | **`0.0086 (0.86%)`** | **Optimal Balance** |

---

## <span  style="color:#309c42ff">3.4 Key Concepts & Diagnostic Insights</span>

### 1. The Overfitting Phenomenon (High Variance)

When a Decision Tree grows without constraints, it creates deeply nested, highly granular decision boundaries tailored to noise and idiosyncratic sample variations in the training set. This manifests as near-perfect training accuracy (`99.88%`) coupled with poor test accuracy (`70.47%`), yielding a severe generalization gap of **29.41%**.

### 2. The Underfitting Phenomenon (High Bias)

Restricting tree depth to `max_depth=1` forces the estimator to make a single binary split across the entire feature space. While the generalization gap is nearly zero (`0.02%`), both train and test accuracies are suppressed at `73.4%`, demonstrating that the model lacks the structural capacity to learn key interaction effects.

### 3. Regularization Through Pre-Pruning

Regularization controls model complexity to achieve optimal bias-variance equilibrium. By constraining the tree with:

- `max_depth=3`: Restricts tree depth to prevent overly specific decision branches.

- `min_samples_leaf=10`: Mandates that leaf nodes contain at least 10 observations, dampening the influence of outliers.

This regularized configuration shrinks the generalization gap from **29.41% down to 0.86%** while raising test set performance to **77.91%**.

---

## <span  style="color:#C39BD3">3.5 Repository Structure</span>

- `Task.ipynb`: Jupyter Notebook containing data loading, data cleaning, train/test split, overfit model evaluation, underfit model evaluation, and regularized model benchmarking.
