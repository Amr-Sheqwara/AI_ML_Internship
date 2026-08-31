
<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 4 — Day 1: Train / Validation / Test Splits

</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

 <b>Day 1 Focus: Three-Way Data Partitioning & Test Set Vault Discipline</b>

Today we focus on foundational model evaluation discipline: eliminating <b>test set data leakage</b>. We partition our dataset into a three-way split (60% Train / 20% Validation / 20% Test), tune hyperparameters strictly against the validation set, and conduct a one-time final audit on the unseen test set.

</blockquote>

---

## <span style="color:#F78BA0">1.1 Overview & Objectives</span>

<blockquote style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

In machine learning, repeatedly checking test data during tuning leaks information into model selection. Today's goal is to establish a leak-free 3-way evaluation workflow.

<b>Key Objectives:</b>

-  Clean and explore the Palmer Penguins dataset (`penguins_size.csv`).
-  Define the Feature Matrix ($X$) and Target Vector ($y$).
-  Implement a 60/20/20 three-way split using two calls to `train_test_split` with `random_state=42`.
-  Apply `StandardScaler` fitted strictly on the training set to prevent feature leakage.
-  Tune the $k$-NN classifier's hyperparameter ($k$) using **Validation Accuracy** only.
-  Evaluate the final chosen model once on the untouched **Test Set** and benchmark against a naive baseline.

</blockquote>

---

## <span style="color:#85C1E9">1.2 Dataset Information</span>

We utilize the **Palmer Archipelago Penguins** dataset for multi-class species classification.

- **Source File:** `../../Data/raw/penguins_size.csv`
- **Target Variable ($y$):** `species` (`Adelie`, `Chinstrap`, `Gentoo`)
- **Features ($X$):** `island`, `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`
- **Instances:** 333 clean rows (after removing missing values and invalid entries).

---

## <span style="color:#F8C471">1.3 Key Concepts & Workflow</span>

### 1. Three-Way Split (`train_test_split`)

```python
from sklearn.model_selection import train_test_split

# 1) Hold out 20% as the final test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 2) Split the remaining 80% into Train (75% -> 60% total) and Validation (25% -> 20% total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)
```

---

## <span style="color:#309c42ff">1.4 Repository Structure</span>

- 📓 `Task.ipynb`: Jupyter Notebook implementing data cleaning, three-way splitting, validation tuning across $k \in [1, 2, 3, 4, 5, 10, 20]$, baseline comparison, and final test evaluation.

