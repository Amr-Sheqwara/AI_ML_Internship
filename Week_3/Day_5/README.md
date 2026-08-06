
<h1 style="color:#9f43c3ff; border-bottom: 3px solid #9f43c3ff; padding-bottom: 8px;">AI & ML Course with BinX — Week 3 — Day 5: Supervised-Learning Mini-Project

</h1>

<blockquote style="border-left: 3px solid #2e1a9aff; padding-left: 12px; margin-left: 0;">

🎯 <b>Day 5 Focus: End-to-End Supervised Learning Pipeline & Model Evaluation</b>

Today we built a complete, leak-free supervised-learning pipeline on the <b>Palmer Penguins Dataset</b> (`penguins_size.csv`) to classify penguin species (`Adelie`, `Gentoo`, `Chinstrap`). The workflow covers Exploratory Data Analysis (EDA), missing value handling, categorical encoding, stratified train/test splitting, feature scaling, baseline benchmarking, model comparison, feature importance analysis, and confusion matrix diagnostics.

</blockquote>

---

## <span style="color:#309c42ff">5.1 Learning Objectives & Workflow</span>

<blockquote style="border-left: 3px solid #FD1D1D; padding-left: 12px; margin-left: 0;">

<b>Key Learning Objectives:</b>

- 🔗 <b>End-to-End Pipeline:</b> Assemble a complete ML pipeline from raw EDA to final model diagnostics.
- 🧹 <b>Leak-Free Preprocessing:</b> Handle missing values, one-hot encode categoricals, and fit `StandardScaler` strictly on training data (`X_train`).
- ✂️ <b>Reproducible Partitioning:</b> Perform a stratified 80/20 train/test split to preserve 3-class target distribution.
- 🤖 <b>Model Comparison:</b> Train and evaluate **Random Forest Classifier** and **k-Nearest Neighbors (k-NN)** via Scikit-Learn.
- 📏 <b>Metric Selection:</b> Benchmark performance using **Macro-averaged F1-Score** against a majority-class `DummyClassifier` baseline.

</blockquote>

---

## <span style="color:#309c42ff">5.2 Dataset Summary</span>

- <b>Source Dataset:</b> Palmer Penguins (`../../Data/penguins_size.csv`)
- <b>Raw Size:</b> 344 rows, 7 columns
- <b>Cleaned Size:</b> 333 rows (11 rows dropped containing missing physical measurements or invalid `sex` entries)
- <b>Target Variable ($y$):</b> `species` — 3 Classes:
  - `Adelie`: 146 (43.8%)
  - `Gentoo`: 119 (35.7%)
  - `Chinstrap`: 68 (20.4%)
- <b>Feature Matrix ($X$):</b> 7 features after One-Hot Encoding (`island_Dream`, `island_Torgersen`, `sex_MALE`, `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`).
- <b>Data Partition:</b> 80/20 Stratified Split (`X_train`: 266 rows / `X_test`: 67 rows, `random_state=42`).

---

## <span style="color:#309c42ff">5.3 Model Benchmarking & Results</span>

All models were evaluated on the held-out, untouched test set ($N = 67$).

| Rank | Model | Key Parameters | Test Accuracy | Test Macro $F_1$-Score | Status / Takeaway |
|:---:|:---|:---|:---:|:---:|:---|
| 🥇 | **Random Forest** | `n_estimators=100`, `random_state=42` | **`1.0000`** | **`1.0000`** | 🏆 **Winner** — Perfect classification across all 3 species on held-out test set. |
| 🥈 | **k-Nearest Neighbors** | `n_neighbors=5` | **`0.9851`** | **`0.9827`** | 📍 **Runner-up** — Misclassified only 1 `Adelie` penguin as `Chinstrap`. |
| 🥉 | **Baseline Model** | `DummyClassifier(most_frequent)` | **`0.4328`** | **`0.2014`** | ⚠️ **Majority Baseline** — Predicts majority class (`Adelie`) for all samples. |

---

## <span style="color:#309c42ff">5.4 Feature Importances (Random Forest)</span>

<blockquote style="border-left: 3px solid #2e1a9aff; padding-left: 12px; margin-left: 0;">

📊 <b>Key Physical Drivers of Penguin Species Classification:</b>

1. <b>`culmen_length_mm`</b> — Primary bill length measurement; strongly differentiates Adelie (short bill) from Chinstrap/Gentoo (long bill).
2. <b>`flipper_length_mm`</b> — Flipper length; key dimension separating Gentoo (large flippers) from Adelie/Chinstrap.
3. <b>`culmen_depth_mm`</b> — Bill depth; separates Gentoo (shallow bill) from Adelie/Chinstrap (deep bill).
4. <b>`body_mass_g`</b> — Overall body mass; secondary indicator for Gentoo size differentiation.
5. <b>Island & Sex features</b> — Minor contribution compared to primary anatomical measurements.

</blockquote>

---

## <span style="color:#309c42ff">5.5 Key Pipeline & Methodology Takeaways</span>

1. <b>Data Leakage Prevention:</b> `StandardScaler.fit_transform()` was executed strictly on `X_train`. `X_test` was processed using `.transform()` only, preserving true test-set isolation.
2. <b>Stratified Splitting:</b> Using `stratify=y` ensured minority class representation (`Chinstrap` ~20%) remained constant across train and test partitions.
3. <b>Metric Selection:</b> Macro-averaged $F_1$-score was selected to treat all 3 classes with equal weight regardless of sample size imbalance.
4. <b>Model Suitability:</b> Random Forest outperformed distance-based k-NN due to its ability to construct orthogonal decision boundaries across physical measurement thresholds.

---

## <span style="color:#309c42ff">5.6 Repository Structure</span>

- 📓 `Task.ipynb`: Complete, executed Jupyter Notebook with EDA, preprocessing, scaling, baseline evaluation, model training, feature importances plot, confusion matrix, and full narrative.
