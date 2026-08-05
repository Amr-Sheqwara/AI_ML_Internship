<h1  style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">

AI & ML Course with BinX — Week 3 — Day 1: Introduction to Scikit-Learn & Data Splitting

</h1>

<blockquote  style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

🐧 <b>Day 1 Focus: Machine Learning Workflow & Train-Test Partitioning</b>

Today marks the beginning of our practical Machine Learning workflow using <b>Scikit-Learn</b>. We focus on foundational data preparation steps: loading tabular data, separating feature matrices ($X$) from target vectors ($y$), and partitioning data into independent training and evaluation sets to prevent data leakage.

</blockquote>

---

## <span style="color:#F78BA0">1.1 Overview & Objectives</span>

<blockquote  style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

In supervised machine learning, models learn generalizable patterns from training data to make accurate predictions on unseen data. Today's goal is to establish a robust, leak-free data splitting pipeline.

<b>Key Objectives:</b>

- 📂 Load and explore the Palmer Penguins dataset (`penguins_size.csv`).

- 🎯 Separate the <b>Feature Matrix ($X$)</b> and <b>Target Vector ($y$)</b>.

- ✂️ Apply `train_test_split` from <code>sklearn.model_selection</code> with an 80/20 train-test ratio.

- 🔒 Set <code>random_state=42</code> for experimental reproducibility.

- 📊 Inspect shape consistency across training and testing partitions.

</blockquote>

---

## <span style="color:#85C1E9">1.2 Dataset Information</span>

We utilize the **Palmer Penguins** dataset, an industry-standard alternative to Iris for classification and regression benchmarking.

-  <b>Source File:</b> `../../Data/penguins_size.csv`

-  <b>Features ($X$):</b> `island`, `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`

-  <b>Target ($y$):</b> `species` (Adelie, Chinstrap, Gentoo)

---

## <span style="color:#F8C471">1.3 Key Concepts & Workflow</span>

### 1. Feature Matrix ($X$) vs. Target Vector ($y$)

In Scikit-Learn conventions:

-  **$X$ (2D DataFrame/Matrix):** Contains input features used by the model to learn predictors.

-  **$y$ (1D Series/Array):** Contains the ground-truth target variable to predict.

### 2. Train-Test Split (`train_test_split`)

```python

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size=0.2, random_state=42

)

```

-  **`test_size=0.2`:** Allocates 80% of data for model training and 20% for final evaluation.

-  **`random_state=42`:** Ensures deterministic shuffling for reproducible results across runs.

---

## <span style="color:#5DADE2">1.4 Repository Contents</span>

- 📓 `Task.ipynb`: Jupyter Notebook implementing data loading, feature/target separation, and `train_test_split` on `penguins_size.csv`.