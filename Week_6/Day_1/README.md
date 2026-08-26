
<h1  style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 6 — Day 1: Sprint 1 Kickoff & Baseline Model
</h1>

<blockquote  style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 1 Focus: Sprint 1 Planning, Neural Network Foundations & Establishing the Classical Baseline Benchmark</b>

Today marks the official kickoff of Phase 3 (Applied Capstone Project) and Sprint 1. In alignment with agile sprint delivery, all modeling tasks are integrated directly into the central Capstone Project repository rather than isolated task notebooks. Today we formalize our Sprint 1 backlog, audit the Cardiac Patient Monitoring dataset (<code>heart.csv</code>), implement a leak-free preprocessing pipeline with physiological data cleaning, train a classical Logistic Regression baseline model, and establish the benchmark performance metrics that every deep learning model built this week must beat.

</blockquote>

---

## <span style="color:#F78BA0">1.1 Overview & Key Objectives</span>

<blockquote  style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

In Sprint 1, establishing a baseline first is essential. Classical ML algorithms often perform strongly on tabular data; therefore, a neural network's score is meaningless without a baseline benchmark. The neural network only earns its place in the clinical pipeline if it demonstrates empirical superiority over the baseline model under identical evaluation conditions.

<b>Key Objectives:</b>

- Confirm the Sprint 1 goal and prioritize backlog tasks (dataset finalization, EDA, and baseline modeling).

- Understand neural network architecture: single neurons as weighted sums ($z = \mathbf{w} \cdot  \mathbf{x} + b$), biases, non-linear activations ($\hat{y} = \sigma(z)$), and layer structures (input, hidden, output).

- Address physiological data anomalies ($0\text{ mm Hg}$ blood pressure and unrecorded $0\text{ mg/dl}$ cholesterol) via median imputation.

- Construct a leak-free Scikit-Learn <code>ColumnTransformer</code> and <code>Pipeline</code>.

- Evaluate the Logistic Regression baseline on a held-out 20% test split ($N = 184$) and record benchmark metrics.

- Structure the Sprint 1 repository workflow, centralizing deliverables in the capstone project.

</blockquote>

---

## <span style="color:#85C1E9">1.2 Dataset Information & Physiological Data Cleaning</span>

We utilize the **Cardiac Patient Monitoring Dataset** for diagnostic classification:

-  **Source File:**  `../../Projects/cardiac_monitoring_project/Data/heart.csv`

-  **Cohort Size:** 918 patient observations

-  **Target Variable (`HeartDisease`):** Binary diagnostic classification (`0` = Normal, `1` = Heart Disease present; 55.3% positive cases)

-  **Feature Space:** 11 clinical, demographic, and exercise electrocardiogram (ECG) markers

### Data Quality Treatment:

1.  **`RestingBP == 0` (1 record):** Blood pressure cannot be $0\text{ mm Hg}$ in a living patient; replaced with `np.nan`.

2.  **`Cholesterol == 0` (172 records):** Structural missingness (unmeasured cholesterol in Hungarian center records); replaced with `np.nan`.

3.  **Leak-Free Imputation:** Handled within the modeling pipeline using `SimpleImputer(strategy='median')` fit strictly on the training set.

---

## <span style="color:#F8C471">1.3 Neural Network Architectural Foundations</span>

  

### 1. The Artificial Neuron

A single neuron computes a linear combination of its inputs (the Week 2 matrix dot product) plus a bias, passed through an activation function:

$$z = \mathbf{w} \cdot  \mathbf{x} + b = \sum_{i=1}^n w_i x_i + b$$

$$\hat{y} = \sigma(z) = \sigma(\mathbf{w} \cdot  \mathbf{x} + b)$$

### 2. Layers and Deep Representations

-  **Input Layer:** Receives the raw feature vector (one node per transformed feature).

-  **Hidden Layers:** Successively transform inputs into higher-level abstract representations. "Deep" learning denotes stacking multiple hidden layers.

-  **Output Layer:** Generates final task predictions (a single sigmoid unit for binary classification producing $\hat{y} \in (0,1)$).

---

## <span style="color:#309c42ff">1.4 Sprint 1 Baseline Benchmark Results</span>

The classical Logistic Regression baseline was trained on an 80% stratified training split ($n = 734$) and evaluated on the held-out 20% test split ($n = 184$).

| Metric | Baseline Score (Logistic Regression with Imputation) | Benchmark Clinical Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | 0.8859 | 88.59% overall correct diagnostic classifications |
| **Precision** | 0.8857 | 88.57% of predicted cardiac cases are true positives |
| **Recall (Sensitivity)** | 0.9118 | 91.18% of actual heart disease patients detected |
| **F1-Score** | 0.8986 | Harmonic balance between clinical precision and recall |
| **ROC-AUC** | 0.9329 | High discriminative power across diagnostic thresholds |

### Sprint 1 Acceptance Criteria

All neural network architectures developed in **Day 4** and hyperparameter-tuned in **Day 5** must exceed this baseline **F1-Score (0.8986)** and **ROC-AUC (0.9329)** under the exact same test split to be accepted into the project pipeline.

---

## <span style="color:#9f43c3ff">1.5 Sprint Integration & Deliverables</span>

In Week 6, sprint execution is performed directly on the applied capstone project:

-  **Capstone Notebook:** (AI_ML_Internship/Projects/cardiac_monitoring_project/Notebooks/Cardiac_Patient_Monitoring_System.ipynb) 

-  **Baseline Notebook:** (AI_ML_Internship/Projects/cardiac_monitoring_project/Notebooks/Baseline_Model.ipynb).

-  **Sprint Documentation:** Daily documentation files (`README.md`) summarize the theoretical concepts, mathematical derivations, and experimental sprint logs.