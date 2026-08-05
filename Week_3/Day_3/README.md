
<h1  style="color:#9f43c3ff; border-bottom: 3px solid #9f43c3ff; padding-bottom: 8px;">AI & ML Course with BinX — Week 3 — Day 3: Logistic Regression & Classification Metrics

</h1>

<blockquote  style="border-left: 3px solid #2e1a9aff; padding-left: 12px; margin-left: 0;">

🎯 <b>Day 3 Focus: Binary Classification, Probability Squashing, Imbalanced Metrics, Confusion Matrix & AUC-ROC Evaluation</b>

Today we trained a binary classification model using <b>Scikit-Learn</b> (`LogisticRegression`) on the <b>Telco Customer Churn Dataset</b> (`Customer-Churn.csv`). We analyzed why simple accuracy fails on imbalanced target distributions (~73% retained / ~27% churned), constructed $2  \times  2$ Confusion Matrices, computed Precision, Recall, $F_1$-score, evaluated class probabilities (`predict_proba`), plotted ROC curves, and evaluated model discrimination via AUC-ROC.

</blockquote>

---

## <span style="color:#309c42ff">3.1 Overview & Learning Objectives</span>

<blockquote  style="border-left: 3px solid #FD1D1D; padding-left: 12px; margin-left: 0;">

<b>Key Learning Objectives:</b>

- 🤖 <b>Logistic Regression Classifier:</b> Train a binary logistic regression model on customer churn data and extract class probabilities (`predict_proba`).

- ⚠️ <b>Imbalanced Data Paradox:</b> Explain why a naive majority-class baseline achieves ~73.5% accuracy while missing 100% of churned customers.

- 🧩 <b>Confusion Matrix:</b> Deconstruct predictions into True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN).

- ⚖️ <b>Precision, Recall & F1-Score:</b> Compute metrics via `classification_report` and justify why Recall is prioritized in customer churn prevention.

- 📈 <b>AUC-ROC Score:</b> Calculate AUC-ROC and plot ROC curves to evaluate classification discrimination across varying decision thresholds.

</blockquote>

---  

## <span style="color:#309c42ff">3.2 Dataset Information</span>

We utilize the **Telco Customer Churn Dataset**:

-  <b>Source File:</b> `../../Data/Customer-Churn.csv`

-  <b>Dataset Dimensions:</b> 7,043 rows, 21 columns

-  <b>Target Variable ($y$):</b> `Churn` (`1` = Churned / Left service [26.54%], `0` = Retained / Stayed [73.46%])

-  <b>Key Numerical & Categorical Features ($X$):</b> `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`, `OnlineSecurity`, `PaperlessBilling`

-  <b>Preprocessing Applied:</b> Numeric conversion and median imputation for `TotalCharges`, dropping `customerID`, and one-hot encoding categorical variables (`pd.get_dummies(..., drop_first=True)` resulting in 30 feature columns).

---  

## <span style="color:#309c42ff">3.3 Key Results & Empirical Summary</span>  

### 1. Confusion Matrix Breakdown (Test Set: $N = 1,409$)

| Actual \ Predicted | Predicted Retained ($\hat{y} = 0$) | Predicted Churn ($\hat{y} = 1$) | Total Actual |
|---|---|---|---|
| **Actual Retained ($y = 0$)** | **927 (TN)**  *(Correct Stay)* | **108 (FP)**  *(False Alarm)* | 1,035 |
| **Actual Churn ($y = 1$)** | **165 (FN)**  *(Missed Churner)* | **209 (TP)**  *(Caught Churner)* | 374 |

---

### 2. Performance Metrics vs. Baseline

| Metric | Logistic Regression | Naive Baseline (Always Retained) | Business Interpretation / Addition |
|---|---|---|---|
| **Overall Accuracy** | **80.91%** | 73.46% | **+7.45%** overall accuracy improvement |
| **Churn Precision** | **0.66 (66%)** | 0.00% | Of predicted churners, 66% actually left |
| **Churn Recall** | **0.56 (56%)** | 0.00% | **Caught 56% of all actual churners** (Baseline catches 0%) |
| **Churn $F_1$-Score** | **0.60 (60%)** | 0.00% | Harmonic mean of precision and recall |
| **AUC-ROC Score** | **`0.8423`** | 0.5000 | **Strong class separation ability across thresholds** |

---

## <span style="color:#309c42ff">3.4 Metric Trade-Off & Business Justification</span>

<blockquote  style="border-left: 3px solid #2e1a9aff; padding-left: 12px; margin-left: 0;">

🎯 <b>Which Metric Matters More for Customer Churn: Precision or Recall?</b>

In <b>Customer Churn Prevention</b>, **Recall is prioritized over Precision**:

1.  **Cost of False Negative (FN — Missed Churner):** High. The company loses total Customer Lifetime Value (LTV) and recurring monthly subscription revenue ($70–$100+/month).

2.  **Cost of False Positive (FP — False Alarm):** Low. Sending a promotional retention email, discount voucher, or customer satisfaction survey to a non-churner costs negligible amounts.

3.  **Strategic Decision:** It is far more profitable to lower the decision threshold to capture more actual churners (maximizing **Recall**), even if it slightly increases false alarms (reducing **Precision**).

</blockquote>

---

## <span style="color:#309c42ff">3.5 Repository Contents</span>

- 📓 Task.ipynb: Fully executed Jupyter Notebook with structured Markdown sections, Confusion Matrix heatmap, ROC Curve visualization, and metric interpretations.