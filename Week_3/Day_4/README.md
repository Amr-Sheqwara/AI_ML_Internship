
<h1 style="color:#9f43c3ff; border-bottom: 3px solid #9f43c3ff; padding-bottom: 8px;">AI & ML Course with BinX — Week 3 — Day 4: Trees, Forests, SVMs & k-NN — Classifier Comparison

</h1>

<blockquote style="border-left: 3px solid #2e1a9aff; padding-left: 12px; margin-left: 0;">

🌲 <b>Day 4 Focus: Decision Trees, Random Forests, Support Vector Machines, k-Nearest Neighbors & Fair Model Benchmarking</b>

Today we trained and evaluated four distinct classification models (<b>Decision Tree</b>, <b>Random Forest</b>, <b>Support Vector Machine (SVM)</b>, and <b>k-Nearest Neighbors (k-NN)</b>) on the <b>Telco Customer Churn Dataset</b> (`Customer-Churn.csv`). We compared the model's performances on the exact same 80/20 train-test split using $F_1$-score, extracted Random Forest feature importances, and analyzed scale invariance versus distance-based limitations.

</blockquote>

---

## <span style="color:#309c42ff">4.1 Overview & Learning Objectives</span>

<blockquote style="border-left: 3px solid #FD1D1D; padding-left: 12px; margin-left: 0;">

<b>Key Learning Objectives:</b>

- 🌴 <b>Decision Trees:</b> Train a rule-based `DecisionTreeClassifier(max_depth=5)` and evaluate interpretability vs. overfitting risk.
- 🌲 <b>Random Forests:</b> Train an ensemble `RandomForestClassifier(n_estimators=100)` and extract Gini feature importances.
- 🎯 <b>Support Vector Machines (SVM):</b> Train an `SVC(kernel="rbf")` and observe the impact of unscaled features on margin calculations.
- 📍 <b>k-Nearest Neighbors (k-NN):</b> Train a `KNeighborsClassifier(n_neighbors=5)` instance-based lazy learner.
- ⚖️ <b>Fair Model Comparison:</b> Benchmark all models on identical split partitions using $F_1$-score ("No Free Lunch" theorem).

</blockquote>

---

## <span style="color:#309c42ff">4.2 Dataset Information</span>

We utilize the **Telco Customer Churn Dataset**:

- <b>Source File:</b> `../../Data/raw/Customer-Churn.csv`
- <b>Dataset Dimensions:</b> 7,043 rows, 21 initial attributes
- <b>Target Variable ($y$):</b> `Churn_Binary` (`1` = Churned [26.54%], `0` = Retained [73.46%])
- <b>Features ($X$):</b> 30 features generated via one-hot encoding (`pd.get_dummies(..., drop_first=True)`), with `TotalCharges` missing values imputed using median imputation.
- <b>Train/Test Partition:</b> Stratified 80/20 train/test split (`X_train`: 5,634 rows / `X_test`: 1,409 rows, `random_state=42`).

---

## <span style="color:#309c42ff">4.3 Key Results & Model Comparison Table</span>

### Model Benchmarking ($F_1$-Score Comparison on Test Set)

| Rank | Model | Key Hyperparameters | Test $F_1$-Score | Performance & Behavior Analysis |
|:---:|:---|:---|:---:|:---|
| 🥇 | **Decision Tree** | `max_depth=5`, `random_state=42` | **`0.5845`** | 🏆 **Winner** — Restricted tree depth prevented overfitting and effectively captured non-linear rules on unscaled data. |
| 🥈 | **Random Forest** | `n_estimators=100`, `random_state=42` | **`0.5501`** | 🌲 **Solid Performer** — Ensemble of 100 decorrelated trees handled tabular features robustly without requiring scaling. |
| 🥉 | **k-Nearest Neighbors** | `n_neighbors=5` | **`0.4939`** | 📍 **Degraded** — Distance metrics were skewed by unscaled numerical ranges and high-dimensional one-hot binary features. |
| 4th | **Support Vector Machine** | `kernel="rbf"`, `probability=True` | **`0.0000`** | ⚠️ **Failed** — Unscaled high-variance features dominated the RBF kernel distance, causing default prediction of 100% majority class (`0`). |

---

## <span style="color:#309c42ff">4.4 Top Feature Importances (Random Forest)</span>

<blockquote style="border-left: 3px solid #2e1a9aff; padding-left: 12px; margin-left: 0;">

📊 <b>Key Drivers of Customer Churn (Mean Decrease in Impurity):</b>

1. <b>`TotalCharges` (19.21%)</b> — Cumulative billing total; reflects customer longevity and total financial engagement.
2. <b>`tenure` (17.47%)</b> — Duration of customer subscription; long-tenured customers exhibit lower switching likelihood.
3. <b>`MonthlyCharges` (16.84%)</b> — Monthly subscription rate; primary price-sensitivity driver for cancellation.
4. <b>`PaymentMethod_Electronic check` (3.88%)</b> — Electronic check users exhibit higher non-automated churn rates.
5. <b>`InternetService_Fiber optic` (3.86%)</b> — High-speed fiber optic users experience higher churn due to price tiering.
6. <b>`Contract_Two year` (3.02%)</b> — Long-term contractual commitments strongly lock in customer retention.

</blockquote>

---

## <span style="color:#309c42ff">4.6 Repository Contents</span>

- 📓 `Task.ipynb`: Fully executed Jupyter Notebook with structured Markdown sections, 4 trained classifiers, F1-score evaluation table, and Random Forest feature importance rankings.
