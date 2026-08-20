
<h1  style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 5 — Day 5: Phase 3 Project Selection & Sprint Planning

</h1>

  

<blockquote  style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

  

<b>Day 5 Focus: Capstone Kickoff, Project Definition of Done, Two-Day Milestone Tracking & Sprint 1 Backlog Execution</b>

  

Today marks the transition from Phase 2 foundational machine learning into the Phase 3 applied capstone implementation. Rather than selecting from generic project prompts, we focus exclusively on our dedicated <b>Cardiac Patient Monitoring System</b>. We formalize the clinical problem statement, define the professional <b>Definition of Done</b>, map the 14-day <b>Two-Day Milestone Plan (M1–M7)</b> against our weekly execution timeline, establish the <b>Sprint 1 Backlog with strict Acceptance Criteria</b>, and document our leak-free modeling workflow.

  

</blockquote>

  

---

  

## <span style="color:#F78BA0">5.1 Overview & Objectives</span>

  

<blockquote  style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

  

Phase 3 is the applied core of the BinX Tech AI & ML Internship Program: building a complete machine learning project end-to-end — from raw clinical data to an evaluated, pipeline-encapsulated model. Week 5 Day 5 closes Phase 2 by finalizing the project scope, defining Sprint 1 deliverables, and aligning our weekly progress with the curriculum's milestone evaluation rubric.

  

<b>Key Objectives:</b>

  

- Select and scope the **Cardiac Patient Monitoring System** as our primary Phase 3 capstone project.

- Formulate the clinical problem statement, input feature modalities, and binary classification objective.

- Define and commit to the **Professional Baseline / Definition of Done (DoD)**.

- Map project progress across the curriculum's **Two-Day Milestone Plan (M1–M7)**.

- Establish the **Sprint 1 Goal, Backlog, Effort Estimation, and Acceptance Criteria**.

- Document the Git feature-branch workflow, leak-free pipeline discipline, and clinical error trade-offs.

  

</blockquote>

  

---

  

## <span style="color:#85C1E9">5.2 Project Selection & Problem Statement</span>

  

### 1. Selected Project: Cardiac Patient Monitoring System

-  **Domain:** Healthcare & Cardiovascular Diagnostics

-  **Primary Data Source:**  `../../Projects/cardiac_monitoring_project/Data/heart.csv`

-  **Cohort Size:** 918 patient observations

-  **Target Variable (`HeartDisease`):** Binary diagnostic classification (`0` = Normal, `1` = Heart Disease present)

-  **Feature Space:** 11 clinical, metabolic, and exercise electrocardiogram (ECG) attributes

  

### 2. Clinical Problem Statement

Cardiovascular disease remains the leading cause of global mortality. Early detection of coronary artery disease and myocardial ischemia enables proactive clinical intervention before irreversible cardiac events occur.

  

The objective of this project is to build an end-to-end machine learning system that:

1. Accurately predicts binary heart disease risk from non-invasive clinical and exercise stress-test markers.

2. Identifies latent patient sub-phenotypes through unsupervised clustering (K-Means) and dimensionality reduction (PCA) without relying on diagnostic labels.

3. Encapsulates all preprocessing, imputation, and feature transformations into a leak-free Scikit-learn Pipeline suitable for reproducible clinical deployment.

  

---

  

## <span style="color:#F8C471">5.3 Professional Baseline & Definition of Done (DoD)</span>

  

Per the program standards outlined in the curriculum, the project must satisfy the following criteria to meet the Definition of Done:

  

| Dimension | Definition of Done Criteria | Current Project Status |
| :--- | :--- | :--- |
| **Reproducibility** | A standalone Jupyter Notebook executing cleanly from start to finish (`Kernel -> Restart & Run All`) without errors or hidden manual steps. | **Completed** (`Cardiac_Patient_Monitoring_System.ipynb`) |
| **Data Quality & Audit** | Documented data dictionary, data type validation, and rigorous handling of structural anomalies (e.g., 172 unrecorded zeros in `Cholesterol`, 1 zero in `RestingBP`). | **Completed** (Median imputation within cross-validation folds) |
| **Exploratory Data Analysis** | Comprehensive statistical summaries, univariate and bivariate distributions, IQR outlier analysis, and correlation heatmaps exported as high-resolution figures. | **Completed** (6 visualizations saved to `Outputs/`) |
| **Supervised Modeling** | Stratified 80/20 train/test split, baseline Logistic Regression, comparison tree-based model (Random Forest), and hyperparameter tuning via `RandomizedSearchCV`. | **Completed** (Stratified 5-fold CV evaluation) |
| **Leak-Free Pipelines** | Encapsulation of all preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) and classification estimators inside Scikit-learn `Pipeline` and `ColumnTransformer`. | **Completed** (`Pipeline` verified with $ROC\text{-}AUC = 0.9301$) |
| **Clinical Evaluation** | Calculation of Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC, with plain-language medical interpretation of False Positives and False Negatives. | **Completed** (Recall prioritized to minimize lethal False Negatives) |
| **Unsupervised Discovery** | Application of K-Means clustering ($k=3$) validated with the Elbow Method and Silhouette Score ($0.2485$), profiling distinct cardiovascular risk phenotypes. | **Completed** (Risk phenotyping documented in Markdown) |
| **Documentation & Git** | Modular directory structure (`Data/`, `Notebooks/`, `Outputs/`), clean `README.md`, frozen `requirements.txt`, and clear Git commit history. | **Completed** |

  

---

  

## <span style="color:#C39BD3">5.4 Two-Day Milestone Plan & Weekly Execution Progress</span>

  

The project follows a structured 14-day milestone framework (7 reviewable two-day blocks). Below is the detailed tracking of each milestone against the weekly schedule:

  

```text

+-------------------------------------------------------------------------------------------------------+

| 14-DAY MILESTONE ROADMAP (M1 - M7) |

+-------------------+---------------------------------------------+-----------------+-------------------+

| Milestone Block | Core Deliverables & Required Output | Review Gate | Completion Status |

+-------------------+---------------------------------------------+-----------------+-------------------+

| M1 (Days 1 - 2) | Python setup, dataset audit, data dict, | Clean notebook | [COMPLETED] |

| | structural zero anomaly cleaning. | loading dataset | |

+-------------------+---------------------------------------------+-----------------+-------------------+

| M2 (Days 3 - 4) | Descriptive statistics, IQR outlier audit, | EDA findings & | [COMPLETED] |

| | univariate/bivariate Matplotlib charts. | charts saved | |

+-------------------+---------------------------------------------+-----------------+-------------------+

| M3 (Days 5 - 6) | Problem formulation, 80/20 stratified split,| Baseline metric | [COMPLETED] |

| | baseline Logistic Regression classifier. | reproducible | |

+-------------------+---------------------------------------------+-----------------+-------------------+

| M4 (Days 7 - 8) | Random Forest model, RandomizedSearchCV, | Model comparison| [COMPLETED] |

| | 5-fold CV, confusion matrix, F1 evaluation. | & tuning done | |

+-------------------+---------------------------------------------+-----------------+-------------------+

| M5 (Days 9 - 10) | Domain feature engineering, Scikit-learn | Leak-free | [COMPLETED] |

| | ColumnTransformer + Pipeline integration. | pipeline tested | |

+-------------------+---------------------------------------------+-----------------+-------------------+

| M6 (Days 11 - 12) | K-Means clustering (k=3), Elbow analysis, | Unsupervised | [COMPLETED] |

| | Silhouette validation, patient phenotyping. | profiling done | |

+-------------------+---------------------------------------------+-----------------+-------------------+

| M7 (Days 13 - 14) | Final README, requirements.txt, limitations,| End-to-end demo | [FINALIZING] |

| | clinical takeaways, demo preparation. | package ready | |

+-------------------+---------------------------------------------+-----------------+-------------------+

```

  

---

  

## <span style="color:#85C1E9">5.5 Sprint 1 Planning: Backlog, Effort & Acceptance Criteria</span>

  

### 1. Sprint 1 Goal

>  **"Audit and clean the 918-patient clinical cardiac dataset, establish an exploratory data analysis baseline with statistical distributions, and build a leak-free baseline classification model with stratified cross-validation."**

  

### 2. Sprint 1 Task Backlog & Effort Estimation

  

| Task ID | Task Description | Estimated Effort | Acceptance Criteria |
| :--- | :--- | :---: | :--- |
| **TSK-01** | **Environment & Dataset Audit** | 4 Hours | - Virtual environment activated with required packages (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).<br>- `heart.csv` loaded and schema verified (918 rows, 12 columns).<br>- Data dictionary documented in `Data/README.md`. |
| **TSK-02** | **Data Cleaning & Missing Value Strategy** | 4 Hours | - 172 zero entries in `Cholesterol` and 1 zero entry in `RestingBP` flagged as structural missingness.<br>- Median imputation strategy defined for numeric features without calculating test-set statistics upfront. |
| **TSK-03** | **Exploratory Data Analysis (EDA)** | 8 Hours | - Univariate distribution plots generated for continuous and categorical attributes.<br>- Bivariate analysis of clinical features against `HeartDisease`.<br>- IQR outlier detection executed and clinically justified.<br>- High-resolution charts exported to `Outputs/`. |
| **TSK-04** | **Baseline Classification Modeling** | 6 Hours | - Stratified 80/20 train/test split implemented (`random_state=42`).<br>- Baseline Logistic Regression fitted on scaled data.<br>- 5-fold stratified cross-validation executed.<br>- Accuracy, Precision, Recall, F1-Score, and ROC-AUC logged. |
| **TSK-05** | **Feature Branch & Git Workflow** | 2 Hours | - Feature branches created for modular development.<br>- Descriptive commit messages adhering to standard conventions.<br>- Markdown summaries documented in notebook cells and daily README files. |

  

---

  

## <span style="color:#309c42ff">5.6 Clinical Error Trade-Off Analysis</span>

  

In cardiac monitoring systems, model performance evaluation requires understanding the asymmetric costs of classification errors:

  

1.  **False Positives (Type I Error):**

-  **Scenario:** Model predicts heart disease for a patient who is healthy.

-  **Clinical Impact:** Patient undergoes non-invasive secondary testing (stress echocardiogram, CT angiogram) and experiences transient anxiety. There is no risk of acute unmonitored cardiovascular death.

  

2.  **False Negatives (Type II Error):**

-  **Scenario:** Model predicts a healthy status for a patient with active cardiac pathology.

-  **Clinical Impact:** Severe and potentially fatal. The patient is discharged without cardioprotective therapy (statins, beta-blockers, ACE inhibitors, or revascularization), leaving them vulnerable to unmonitored myocardial infarction.

  

3.  **Optimization Strategy:**

- Model tuning and decision threshold selection specifically optimize for **Recall / Sensitivity** on the positive class (`HeartDisease = 1`) to minimize False Negatives while maintaining clinically acceptable Precision.

  

---

  

## <span style="color:#F78BA0">5.7 Repository & Deliverables Summary</span>

  

```text

AI_ML_Internship/

├── Projects/

│ └── cardiac_monitoring_project/

│ ├── Data/

│ │ ├── README.md # Data dictionary & clinical metadata

│ │ └── heart.csv # Clinical dataset (918 records)

│ ├── Notebooks/

│ │ └── Cardiac_Patient_Monitoring_System.ipynb # Full pipeline & modeling notebook

│ ├── Outputs/ # High-resolution charts & figures

│ ├── README.md # Comprehensive capstone project report

│ └── requirements.txt # Frozen project dependencies

└── Week_5/

├── Day_1/ # K-Means clustering & silhouette validation

├── Day_2/ # DBSCAN & hierarchical clustering

├── Day_3/ # PCA dimensionality reduction

├── Day_4/ # t-SNE & Isolation Forest anomaly detection

├── Day_5/

│ └── README.md # Phase 3 planning & Sprint 1 execution roadmap

└── README.md # Week 5 curriculum overview

```