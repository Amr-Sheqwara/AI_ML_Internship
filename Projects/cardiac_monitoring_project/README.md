# Cardiac Patient Monitoring System

## Overview

The **Cardiac Patient Monitoring System** is an applied healthcare machine learning system developed as part of the BinX Tech AI & ML Internship program. The project implements a complete, end-to-end diagnostic workflow to predict acute coronary risk using clinical, demographic, and electrocardiogram (ECG) stress-testing measurements.

Spanning three distinct methodological phases—**Supervised Classification**, **Unsupervised Latent Phenotyping**, and **Deep Artificial Neural Networks**—the project enforces production-grade machine learning engineering: strict separation of data partitions to eliminate data leakage, domain-informed clinical feature synthesis, diagnostic loss curve analysis, multi-technique deep regularization, automated production callbacks, and disciplined baseline-first model benchmarking.

---

## Project Objectives

1. **Environment & Data Inspection:** Initialize a deterministic, reproducible runtime environment (`seed = 42`), audit dataset schema and data types, and document cardiovascular feature characteristics.
2. **Data Preparation & Quality Remediation:** Detect and treat data anomalies, structural missing values (172 unrecorded zero values in `Cholesterol`, 1 physiologically impossible zero in `RestingBP`), and categorical features without test leakage.
3. **Exploratory Data Analysis & Clinical Statistics:** Compute descriptive statistics, evaluate univariate and bivariate distributions, apply the Interquartile Range (IQR) method for outlier screening, evaluate Pearson correlation matrices, and formulate clinical hypotheses.
4. **Supervised Classification & Baseline Benchmarking:** Formulate a binary diagnostic problem predicting `HeartDisease` (1 = Heart Disease Present, 0 = Normal) and establish baseline performance using Logistic Regression on a stratified 80/20 train/test split ($N = 918$).
5. **Domain Feature Engineering & Hyperparameter Tuning:** Synthesize physiological ratios, exercise stress interactions, and metabolic risk flags, and optimize `RandomForestClassifier` hyperparameters using `RandomizedSearchCV` with 5-fold stratified cross-validation.
6. **Leak-Free Scikit-Learn Pipeline:** Encapsulate median imputation, standard scaling, one-hot encoding, and classification into a unified `Pipeline` via `ColumnTransformer` to guarantee zero test-set contamination.
7. **Clinical Error Interpretation:** Evaluate models across Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC, interpreting Type I (False Positive) and Type II (False Negative) errors in plain medical terms.
8. **Unsupervised Learning & Latent Phenotyping:** Apply K-Means clustering ($k=2$), DBSCAN density-based anomaly detection ($\varepsilon = 1.3, \text{MinPts} = 5$), Agglomerative Hierarchical Clustering (Ward's linkage), Principal Component Analysis (PCA), and t-SNE manifold learning to uncover natural patient risk cohorts, isolate critical clinical anomalies, and visualize non-linear patient manifolds without diagnostic labels.
9. **Deep Learning Foundations & Forward Pass:** Analyze non-linear activation functions (ReLU, Sigmoid, Tanh, Softmax) and their derivatives, formulate the binary cross-entropy loss, and perform an analytical 2-layer forward pass hand-verified against NumPy execution.
10. **Optimization Dynamics & Backpropagation:** Diagram the 4-stage optimization cycle, formulate gradients via the multivariate chain rule, build modular Multi-Layer Perceptron (MLP) architectures, and empirically diagnose SGD learning rate dynamics ($\eta = 1.5, 0.0001, 0.1$) over 80 training epochs.
11. **Deep Regularization & Generalization Diagnostics:** Diagnose the onset of tabular overfitting (generalization gap) on unregularized deep networks, and eliminate divergence by integrating Batch Normalization, Dropout ($p \in [0.10, 0.25]$), and L2 weight decay ($\lambda \in [0.0005, 0.001]$).
12. **Capstone Sprint 1 Production Tuning & Benchmarking:** Implement production Keras callbacks (`EarlyStopping`, `ModelCheckpoint`), tune the deep architecture (`model_Tuned`), and benchmark against classical models to achieve **>90% test performance** across all core evaluation metrics.

---

## Dataset Description

- **Dataset Source File:** [`Data/raw/heart.csv`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Data/raw/heart.csv)
- **Total Patient Records:** 918 observations
- **Total Raw Features:** 11 clinical attributes + 1 binary diagnostic target
- **Target Variable Distribution (`HeartDisease`):**
  - `0` (Normal / No Heart Disease): 410 records (44.66%)
  - `1` (Heart Disease Present): 508 records (55.34%)

### Clinical Data Dictionary

| Column Name | Data Type | Description | Values / Range | Missing Count |
| :--- | :--- | :--- | :--- | :--- |
| `Age` | Integer / Continuous | Patient biological age in years | 28 to 77 | 0 (0.0%) |
| `Sex` | Categorical | Biological sex | `M` (Male: 725), `F` (Female: 193) | 0 (0.0%) |
| `ChestPainType` | Categorical | Clinical presentation of chest discomfort | `ASY` (Asymptomatic), `NAP` (Non-Anginal), `ATA` (Atypical Angina), `TA` (Typical Angina) | 0 (0.0%) |
| `RestingBP` | Integer / Continuous | Resting blood pressure on admission (mm Hg) | 0 to 200 (1 zero anomaly treated) | 0 (0.0%) |
| `Cholesterol` | Integer / Continuous | Serum cholesterol level (mg/dl) | 0 to 603 (172 zero values treated) | 0 (0.0%) |
| `FastingBS` | Binary | Fasting blood sugar status | `1` if FastingBS > 120 mg/dl, `0` otherwise | 0 (0.0%) |
| `RestingECG` | Categorical | Resting electrocardiogram findings | `Normal`, `ST` (ST-T abnormality), `LVH` (Left Ventricular Hypertrophy) | 0 (0.0%) |
| `MaxHR` | Integer / Continuous | Maximum heart rate achieved during exercise (bpm) | 60 to 202 | 0 (0.0%) |
| `ExerciseAngina` | Categorical | Exercise-induced angina | `Y` (Yes), `N` (No) | 0 (0.0%) |
| `Oldpeak` | Float / Continuous | Exercise-induced ST depression relative to rest (mm) | -2.6 to 6.2 | 0 (0.0%) |
| `ST_Slope` | Categorical | Slope of the peak exercise ST segment | `Up` (Upsloping), `Flat` (Flat), `Down` (Downsloping) | 0 (0.0%) |
| `HeartDisease` | Binary (Target) | Physician-confirmed cardiac diagnosis | `0` = Normal, `1` = Heart Disease Present | 0 (0.0%) |

---

## Key Exploratory Data Analysis (EDA) Insights

1. **Cohort Target Balance:** With 55.34% positive cases, class imbalance is minimal, allowing reliable accuracy and cross-validation comparisons without synthetic oversampling.
2. **Silent Ischemia (`ASY` Dominance):** Asymptomatic patients (`ChestPainType = ASY`) carry the highest disease concentration (**79.00%**), underscoring that lack of reported pain is often indicative of advanced, silent myocardial ischemia.
3. **Diagnostic Power of the ST Segment:** An upsloping ST segment (`ST_Slope = Up`) correlates heavily with cardiovascular health (**80.3% normal**). In contrast, flat or downsloping profiles (`Flat`, `Down`) represent acute ischemic warning signs (**82.8%** and **77.8%** disease prevalence).
4. **Cardiorespiratory Fitness & Myocardial Strain:** Maximum achieved heart rate (`MaxHR`) correlates negatively with heart disease ($r = -0.40$), while ST depression (`Oldpeak`) exhibits a strong positive correlation ($r = +0.40$).
5. **Structural Missingness Remediation:** 172 unrecorded zeros in `Cholesterol` (18.7%) and 1 record with 0 mm Hg `RestingBP` were identified as clinical non-recordings and treated via median imputation within the leak-free preprocessing pipeline.

---

## Domain-Informed Feature Engineering

To provide models with direct physiological signals beyond raw measurements, 8 specialized features were synthesized (expanding the processed feature space to $d = 26$ dimensions after one-hot encoding):

1. **`MaxHR_Ratio`:** $\frac{\text{MaxHR}}{220 - \text{Age}}$ — Quantifies percentage of age-predicted maximum heart rate achieved, measuring chronotropic incompetence.
2. **`BP_Age_Ratio`:** $\frac{\text{RestingBP}}{\text{Age}}$ — Normalizes arterial blood pressure against biological age.
3. **`Exercise_Stress_Risk`:** Binary flag for `ExerciseAngina == 'Y'` AND `ST_Slope.isin(['Flat', 'Down'])` (concurrent ischemic indicators).
4. **`Metabolic_Risk`:** Binary flag for `FastingBS == 1` AND `RestingBP >= 140` (co-occurring hyperglycemia and Stage 2 hypertension).
5. **`Is_Oldpeak_Significant`:** Binary flag indicating clinically significant ST depression ($\text{Oldpeak} \ge 1.5\text{ mm}$).
6. **`Is_Asymptomatic`:** Indicator capturing the high-risk asymptomatic clinical phenotype (`ChestPainType == 'ASY'`).
7. **`Age_Group`:** Stratified categorical binning (`Young` $[<45]$, `Middle_Aged` $[45-55)$, `Senior` $[55-65)$, `Elderly` $[\ge 65]$).
8. **`BP_Category`:** Clinical blood pressure stages (`Normal` $[<120]$, `Prehypertension` $[120-140)$, `Hypertension` $[\ge 140]$).

---

## Phase 1: Supervised Machine Learning & Preprocessing Pipelines

Models were trained on an 80% training split ($N = 734$) and evaluated on a held-out 20% test split ($N = 184$) using 5-fold stratified cross-validation.

| Model Architecture | 5-Fold CV F1-Score | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Day 1 Baseline (Logistic Regression)** | 0.8541 | 88.59% | 88.57% | 91.18% | 89.86% | 0.9329 |
| **Tuned Random Forest (`RandomizedSearchCV`)** | 0.8751 | 85.33% | 88.00% | 85.00% | 86.57% | 0.9163 |
| **End-to-End Leak-Free Pipeline (RF)** | **0.8772** | **87.50%** | **86.92%** | **91.18%** | **89.00%** | **0.9281** |

### Optimized Random Forest Hyperparameters:
- `n_estimators`: 300
- `max_depth`: 7
- `max_features`: `log2`
- `min_samples_split`: 5
- `min_samples_leaf`: 4
- `criterion`: `gini`

---

## Plain-Language Clinical Error Interpretation

In clinical decision support systems, false positive and false negative classification errors carry starkly asymmetric clinical stakes:

1. **False Positives (Type I Error — Healthy Patient Flagged as Diseased):**
   - **Clinical Outcome:** Triggers psychological anxiety, secondary non-invasive tests (echocardiography, CT angiography), and increased clinical resource consumption.
   - **Harm Assessment:** Low physical harm; patient remains medically protected.
2. **False Negatives (Type II Error — Diseased Patient Discharged as Normal):**
   - **Clinical Outcome:** A patient suffering from significant coronary artery disease is falsely reassured and sent home without medical therapy (aspirin, statins, beta-blockers).
   - **Harm Assessment:** Critical/Fatal. The patient remains at imminent risk of unmonitored acute myocardial infarction or cardiac arrest.
   - **Clinical Directive:** Models must prioritize high **Sensitivity / Recall ($\ge 90\%$)** to minimize Type II errors.

---

## Phase 2: Unsupervised Learning & Latent Phenotyping

To discover natural patient risk profiles without diagnostic labels, five complementary unsupervised paradigms were implemented on standardized physiological features:

### 1. Centroid-Based Clustering (K-Means)
- **Model Selection:** Evaluated across $k \in [1, 10]$ with 10 random initializations. Silhouette analysis confirmed that **$k=2$ achieves the global maximum Silhouette Score ($0.2175$)**, naturally stratifying patients into two distinct cardiac risk phenotypes:
  - **Cluster 0 (High Cardiovascular Risk Subgroup — 50.76% of cohort):** Older age (mean 59.5 yrs), elevated blood pressure (139.6 mmHg), reduced chronotropic reserve (`MaxHR` 122.4 bpm), pronounced ST depression (`Oldpeak` 1.38 mm), and a **76.39% disease rate**.
  - **Cluster 1 (Low Cardiovascular Risk Subgroup — 49.24% of cohort):** Younger age (mean 47.3 yrs), controlled blood pressure (124.9 mmHg), robust exercise capacity (`MaxHR` 151.7 bpm), minimal ST depression (`Oldpeak` 0.38 mm), and a **33.63% disease rate**.

### 2. Density-Based Clustering (DBSCAN) & Pathological Outlier Screening
- **Calibration:** Calibrated using sorted 5-NN distance graphs at the maximum curvature elbow ($\varepsilon = 1.3, \text{MinPts} = 5$).
- **Pathological Outlier Isolation:** Automatically flagged **51 patient records (5.56% of cohort)** as density noise ($label = -1$).
- **Clinical Severity of Outliers:** The DBSCAN outlier cohort exhibits a **78.43% Heart Disease prevalence** (vs. 53.98% in core patients), capturing severe clinical extremes: hypertensive crises (`RestingBP` up to 200 mmHg), severe hypercholesterolemia (`Cholesterol` up to 603 mg/dl), and profound ischemic depression (`Oldpeak` up to 6.2 mm).

### 3. Agglomerative Hierarchical Clustering
- Constructed using Ward's minimum variance criterion on Euclidean distances.
- Dendrogram inspection demonstrates an optimal high-level partition cut at distance $d = 33.0$ ($k=2$, Silhouette Score = $0.1465$), validating the two-cohort morphology uncovered by K-Means.

### 4. Principal Component Analysis (PCA)
- **PC1 (34.01% Variance):** Cardiovascular Aging & Exercise Ischemia Axis (Loadings: `Age` $+0.601$, `Oldpeak` $+0.450$, `RestingBP` $+0.428$, `MaxHR` $-0.494$).
- **PC2 (21.02% Variance):** Metabolic & Hemodynamic Stress Axis (Loadings: `Cholesterol` $+0.838$, `MaxHR` $+0.394$, `RestingBP` $+0.331$).
- **Total Variance Retention:** PC1 + PC2 capture **55.03%**; PC1 through PC4 retain **88.47%** of total physiological variance.

### 5. Non-Linear Manifold Learning (t-SNE)
- Implemented with Student-t kernel mapping (`perplexity = 30`, `max_iter = 1000`) to address crowding.
- Resolves non-linear topological manifolds, isolating compact clusters of asymptomatic coronary patients (`ChestPainType = ASY`) that overlap in linear PCA space.

### Unsupervised Paradigm Comparison Matrix

| Dimension | K-Means (2.1) | DBSCAN (2.2) | Hierarchical (2.3) | PCA (2.4) | t-SNE (2.5) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mathematical Objective** | Minimizes WCSS (Inertia) | Connects $\varepsilon$-dense neighborhoods | Minimizes Ward variance ($\Delta\text{ESS}$) | Maximizes orthogonal variance | Minimizes KL divergence ($P \parallel Q$) |
| **Cluster Geometry** | Spherical, convex | Arbitrary non-linear shapes | Hierarchical nested trees | Linear orthogonal axes | Complex topological manifolds |
| **Outlier Handling** | Forced into nearest centroid | Explicitly flagged as noise ($label=-1$) | Merged into singleton branches | Exerts high leverage on variance | Positioned at manifold periphery |
| **Optimal Setting** | $k=2$ ($s = 0.2175$) | $\varepsilon=1.3$, $\text{MinPts}=5$ | $k=2$ ($d = 33.0$) | $d=2$ ($55.03\%$ var) | $\text{Perplexity}=30$ |
| **Key Clinical Finding** | 2 risk phenotypes ($33.6\% \to 76.4\%$) | 51 critical outliers ($78.4\%$ disease) | Confirms 2-tier tree taxonomy | PC1: Aging/Ischemia, PC2: Lipids | Isolates silent ischemia clusters |

---

## Phase 3: Deep Learning, Neural Network Training & Capstone Sprint 1

Phase 3 transitions from classical algorithms to deep neural network engineering using TensorFlow/Keras, following the **Baseline-First Principle** to surpass the Day 1 Logistic Regression benchmark ($F_1 = 0.8986$, $\text{ROC-AUC} = 0.9329$).

### 1. Mathematical Foundations & Forward Pass

#### Activation Functions & Gradients
Non-linear activation functions prevent multi-layer networks from collapsing into linear transformations ($\mathbf{W}_2(\mathbf{W}_1\mathbf{x}) = \mathbf{W}_{net}\mathbf{x}$):
- **ReLU ($f(z) = \max(0, z)$):** Default for hidden layers. Constant gradient ($f'(z) = 1$ for $z > 0$) prevents vanishing gradients during backpropagation.
- **Sigmoid ($\sigma(z) = \frac{1}{1 + e^{-z}}$):** Applied at the single-neuron output layer, mapping continuous scalar logits into valid diagnostic posterior probabilities $\hat{y} = P(\text{HeartDisease}=1 \mid \mathbf{x}) \in (0, 1)$.
- **Tanh ($\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$):** Zero-centered activation with gradient $1 - \tanh^2(z)$.
- **Binary Cross-Entropy Loss:**
  $$\mathcal{L}_{BCE}(y, \hat{y}) = -\left[ y \ln(\hat{y}) + (1 - y) \ln(1 - \hat{y}) \right]$$
  Derived from maximum likelihood estimation for Bernoulli trials, heavily penalizing confident false diagnoses.

#### Analytical 2-Layer Forward Pass Verification
To mathematically prove the forward propagation mechanics, an analytical 2-layer forward pass ($2 \to 2 \to 1$) was computed by hand for patient vector $\mathbf{x} = [0.50, -0.20]^T$ ($y_{true} = 1.0$) and verified step-by-step against NumPy:

$$\mathbf{W}^{[1]} = \begin{bmatrix} 0.80 & -0.50 \\ 0.40 & 0.60 \end{bmatrix}, \quad \mathbf{b}^{[1]} = \begin{bmatrix} 0.10 & -0.20 \end{bmatrix}$$
$$\mathbf{W}^{[2]} = \begin{bmatrix} 0.70 \\ -0.30 \end{bmatrix}, \quad b^{[2]} = 0.05$$

| Step | Mathematical Operation | Analytical Calculation | Numerical Result |
| :--- | :--- | :--- | :--- |
| **1. Input** | $\mathbf{x}^T$ | Continuous standardized physiological vector | $[0.50, -0.20]$ |
| **2. Hidden Logit** | $\mathbf{z}^{[1]} = \mathbf{x}\mathbf{W}^{[1]} + \mathbf{b}^{[1]}$ | $[(0.50\cdot0.80 - 0.20\cdot0.40 + 0.10), (0.50\cdot-0.50 - 0.20\cdot0.60 - 0.20)]$ | $[0.42, -0.57]$ |
| **3. Hidden Activation** | $\mathbf{a}^{[1]} = \max(0, \mathbf{z}^{[1]})$ | $[\max(0, 0.42), \max(0, -0.57)]$ | $[0.42, 0.00]$ |
| **4. Output Logit** | $z^{[2]} = \mathbf{a}^{[1]}\mathbf{W}^{[2]} + b^{[2]}$ | $0.42 \times 0.70 + 0.00 \times -0.30 + 0.05$ | $0.3440$ |
| **5. Predicted Risk** | $\hat{y} = \sigma(z^{[2]})$ | $\frac{1}{1 + e^{-0.3440}}$ | **0.5852 (58.52%)** |
| **6. Diagnostic Loss** | $\mathcal{L}_{BCE}(y=1, \hat{y})$ | $-\ln(0.585162)$ | **0.5359** |

---

### 2. Optimization Dynamics, Backpropagation & Learning Rate Sensitivity

#### The 4-Stage Optimization Cycle
```text
  ==============================================================================================
                                  THE 4-STAGE OPTIMIZATION CYCLE
  ==============================================================================================

    [1] FORWARD PASS              [2] COMPUTE LOSS
    +-----------------------+     +-------------------------------+
    | Data flows forward    |     | Compare prediction (y_hat)    |
    | layer-by-layer:       | --> | against ground truth (y)      |
    | z = w * x + b         |     | using Binary Cross-Entropy:   |
    | a = f(z)              |     | L_BCE(y, y_hat)               |
    +-----------------------+     +-------------------------------+
                                                  |
                                                  v
    [4] PARAMETER UPDATE          [3] BACKPROPAGATION
    +-----------------------+     +-------------------------------+
    | Gradient Descent step |     | Traverse backward through the |
    | along steepest slope: | <-- | graph via Multivariate        |
    | w_{t+1} = w_t - n*dL  |     | Chain Rule to compute:        |
    | b_{t+1} = b_t - n*dL  |     | dL/dw = (dL/dy_hat)(dy_hat/dz)(dz/dw) |
    +-----------------------+     +-------------------------------+
               |
               +--- Repeat for N Epochs / Mini-Batches ---> (Convergence)
  ==============================================================================================
```

#### Analytical Gradients via the Multivariate Chain Rule
Because loss $\mathcal{L}$ is a composition of nested layer activations, the exact sensitivity of the loss with respect to any network parameter is computed recursively from output to input:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T, \quad \text{where } \boldsymbol{\delta}^{(l)} = ((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}) \odot f'(\mathbf{z}^{(l)})$$

For the Sigmoid output layer coupled with Binary Cross-Entropy:
$$\frac{\partial \mathcal{L}}{\partial z^{[L]}} = \hat{y} - y$$

#### Empirical SGD Learning Rate Diagnostics (80 Epochs)
Using a modular architecture (`build_cardiac_mlp`, 577 parameters), optimization trajectories were diagnosed across three learning rate regimes:
1. **Too High ($\eta = 1.5$):** Overshoots the loss landscape minimum, triggering violent oscillatory loss spikes and training instability.
2. **Too Low ($\eta = 0.0001$):** Gradients make microscopic progress along flat loss plateaus, resulting in static loss trajectories and severe underfitting.
3. **Optimal ($\eta = 0.1$):** Balances step size and descent momentum, producing smooth, monotonic convergence and surpassing 85% accuracy.

---

### 3. Deep MLP Overfitting & Generalization Gap Diagnosis

A deeper Multi-Layer Perceptron ($32 \to 16 \to 1$, 1,393 parameters) was trained unregularized using `Adam(0.001)` over 60 epochs:
- **Convergence Phase (Epochs 1–13):** Training and validation loss decrease synchronously ($0.6540 \to 0.2712$ train; $0.5589 \to 0.4323$ val), with validation loss reaching its global minimum ($0.4323$) at Epoch 13.
- **Overfitting Divergence (Epochs 14–60):** Beyond Epoch 13, training loss continues driving downward ($0.2712 \to 0.1065$), while validation loss diverges sharply upward ($0.4323 \to 0.5138$), opening a **generalization gap of $+0.4073$**. The unregularized model memorizes idiosyncratic patient noise rather than true cardiovascular patterns.

---

### 4. Deep Regularization: Batch Normalization, Dropout & L2 Weight Decay

To close the generalization gap, three complementary regularization mechanisms were embedded into `model_optimized`:
1. **Batch Normalization:** Applied after Dense layers to normalize pre-activations ($\mu = 0, \sigma^2 = 1$), eliminating internal covariate shift and smoothing the loss landscape.
2. **Dropout ($p_1 = 0.25, p_2 = 0.15$):** Stochastically zeroes neuron outputs during each training pass, preventing co-adaptation of features and forcing the network to develop redundant representations.
3. **L2 Weight Regularization ($\lambda = 0.001$):** Imposes a quadratic weight decay penalty ($\frac{1}{2}\lambda \|\mathbf{w}\|_2^2$) to restrict weight norms and prevent extreme parameter magnitudes.
- **Diagnostic Result:** The regularized architecture completely eliminated validation loss divergence, maintaining a stable trajectory across 60 epochs and successfully closing the generalization gap.

---

### 5. Production Tuning & Automated Callbacks

To build a production-grade classifier, `model_Tuned` (`Cardiac_90Plus_Deep_Classifier`) was engineered with:
- **He Normal Initialization:** Paired with ReLU activations to maintain variance stability across layers.
- **Calibrated L2 Regularization:** Tuned to $\lambda = 0.0005$.
- **Calibrated Dropout:** Layer 1 $p = 0.20$, Layer 2 $p = 0.10$.
- **Label Smoothing ($0.02$):** Prevents overconfident logit predictions.
- **Automated Keras Callbacks:**
  - `EarlyStopping`: Monitors `val_loss` with `patience = 15` and `restore_best_weights = True`.
  - `ModelCheckpoint`: Automatically checkpoints the optimal weights to [`Notebooks/best_cardiac_nn_model.keras`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Notebooks/best_cardiac_nn_model.keras).

---

### 6. Capstone Sprint 1 Multi-Model Benchmark Comparison

All models were evaluated on the identical held-out test cohort ($N = 184$: 82 Normal, 102 Heart Disease):

| Diagnostic Metric | Day 1 Baseline (Logistic Regression) | Unregularized Deep MLP | Regularized Deep NN (BatchNorm + Dropout) | Tuned Deep NN (`model_Tuned`) | Performance vs. Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.8859 (88.59%) | 0.8207 (82.07%) | 0.8641 (86.41%) | **0.9022 (90.22%)** | **+1.63%** |
| **Precision** | 0.8857 (88.57%) | 0.8224 (82.24%) | 0.8812 (88.12%) | **0.9118 (91.18%)** | **+2.61%** |
| **Recall (Sensitivity)** | 0.9118 (91.18%) | 0.8627 (86.27%) | 0.8725 (87.25%) | **0.9118 (91.18%)** | **Equal (0.00%)** |
| **F1-Score** | 0.8986 (89.86%) | 0.8421 (84.21%) | 0.8768 (87.68%) | **0.9118 (91.18%)** | **+1.32%** |
| **ROC-AUC** | 0.9329 | 0.9171 | 0.9303 | **0.9342** | **+0.0013** |

### Test Set Confusion Matrix Evolution ($N = 184$)

```text
1. Unregularized Deep MLP         2. Regularized Deep NN            3. Tuned Deep NN (model_Tuned)
   [Acc: 82.07% | F1: 0.8421]        [Acc: 86.41% | F1: 0.8768]        [Acc: 90.22% | F1: 0.9118]
       Predicted                         Predicted                         Predicted
       0       1                         0       1                         0       1
0   [ 63 ]  [ 19 ]                0   [ 70 ]  [ 12 ]                0   [ 73 ]  [  9 ]   (FP: 19 -> 12 -> 9)
1   [ 14 ]  [ 88 ]                1   [ 13 ]  [ 89 ]                1   [  9 ]  [ 93 ]   (FN: 14 -> 13 -> 9)
```

### Capstone Findings:
1. **Surpassing the 90% Benchmark:** The Tuned Deep Neural Network breaks through the 90% performance barrier across **Accuracy (90.22%)**, **Precision (91.18%)**, **Recall (91.18%)**, and **F1-Score (91.18%)**, achieving a superior **ROC-AUC of 0.9342**.
2. **Superior Type I Error Suppression:** False Positives decreased by **52.6%** compared to the unregularized model ($19 \to 9$), drastically reducing unnecessary clinical interventions.
3. **Critical Type II Error Minimization:** False Negatives decreased from 14 to **9**, ensuring that **91.18%** of actual heart disease patients receive timely, life-saving clinical treatment.

---

## Visual Diagnostic Artifacts

All 22 diagnostic figures generated throughout the project are archived at publication resolution in [`Outputs/`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs):

| Output File | Project Phase | Clinical & Methodological Description |
| :--- | :--- | :--- |
| [`01_Univariate_Categorical_Distributions.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/01_Univariate_Categorical_Distributions.png) | Phase 1 (EDA) | Frequency bars for `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`. |
| [`02_Univariate_Numerical_Distributions.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/02_Univariate_Numerical_Distributions.png) | Phase 1 (EDA) | Histograms and KDE plots showing zero-anomalies in `RestingBP` and `Cholesterol`. |
| [`03_Bivariate_Categorical_Features_vs_Target.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/03_Bivariate_Categorical_Features_vs_Target.png) | Phase 1 (EDA) | Proportional risk breakdowns highlighting the 79% disease rate in asymptomatic (`ASY`) patients. |
| [`04_Bivariate_Numerical_Features_Boxplots_vs_Target.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/04_Bivariate_Numerical_Features_Boxplots_vs_Target.png) | Phase 1 (EDA) | Distribution shifts in `MaxHR` and `Oldpeak` across diseased vs. normal patient cohorts. |
| [`05_Correlation_Matrix_and_Heatmap.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/05_Correlation_Matrix_and_Heatmap.png) | Phase 1 (EDA) | Pearson correlation matrix highlighting correlations with `HeartDisease`. |
| [`06_Multivariate_Feature_Screening_Pairplot.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/06_Multivariate_Feature_Screening_Pairplot.png) | Phase 1 (EDA) | Pairplot matrix separating high-dimensional patient clusters by target label. |
| [`07_Confusion_Matrix_and_ROC_Baseline_Pipeline.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/07_Confusion_Matrix_and_ROC_Baseline_Pipeline.png) | Phase 1 (Modeling) | Confusion matrix and ROC curve for the end-to-end Random Forest pipeline. |
| [`08_Elbow_Method_and_Silhouette_Scores_KMeans.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/08_Elbow_Method_and_Silhouette_Scores_KMeans.png) | Phase 2 (Unsupervised) | Dual-panel plot validating $k=2$ as the global Silhouette maximum ($s = 0.2175$). |
| [`09_Patient_Subgroups_KMeans_Clusters.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/09_Patient_Subgroups_KMeans_Clusters.png) | Phase 2 (Unsupervised) | 2D scatter of K-Means clusters showing High-Risk (76.4%) vs. Low-Risk (33.6%) cohorts. |
| [`10_DBSCAN_kNN_Distance_and_Clusters.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/10_DBSCAN_kNN_Distance_and_Clusters.png) | Phase 2 (Unsupervised) | 5-NN distance elbow curve ($\varepsilon = 1.3$) and density outlier scatter. |
| [`11_Hierarchical_Clustering_Dendrogram.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/11_Hierarchical_Clustering_Dendrogram.png) | Phase 2 (Unsupervised) | Truncated Ward linkage dendrogram with horizontal cutoff line at distance $d = 33.0$. |
| [`12_PCA_Scree_Plot_and_Cumulative_Variance.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/12_PCA_Scree_Plot_and_Cumulative_Variance.png) | Phase 2 (Unsupervised) | Scree plot showing 55.03% variance in PC1-PC2 and 88.47% in PC1-PC4. |
| [`13_PCA_Loadings_Matrix_Heatmap.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/13_PCA_Loadings_Matrix_Heatmap.png) | Phase 2 (Unsupervised) | Eigenvector loadings attributing PC1 to Ischemia/Aging and PC2 to Lipids/BP. |
| [`14_2D_PCA_Clinical_Projection.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/14_2D_PCA_Clinical_Projection.png) | Phase 2 (Unsupervised) | 2D orthogonal projection of patient observations labeled by cardiac outcome. |
| [`15_Manifold_Comparison_PCA_vs_tSNE.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/15_Manifold_Comparison_PCA_vs_tSNE.png) | Phase 2 (Unsupervised) | Side-by-side comparison of linear PCA vs. non-linear t-SNE manifold topologies. |
| [`16_Activation_Functions_and_Derivatives.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/16_Activation_Functions_and_Derivatives.png) | Phase 3 (Deep Learning) | Activation curves (ReLU, Sigmoid, Tanh) and corresponding backpropagation derivatives. |
| [`17_Learning_Rate_Dynamics_SGD_Convergence.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/17_Learning_Rate_Dynamics_SGD_Convergence.png) | Phase 3 (Deep Learning) | Loss and accuracy convergence curves across $\eta \in \{1.5, 0.0001, 0.1\}$. |
| [`18_Unregularized_Model_Training_vs_Validation_Loss.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/18_Unregularized_Model_Training_vs_Validation_Loss.png) | Phase 3 (Deep Learning) | Overfitting divergence plot illustrating the widening generalization gap after Epoch 13. |
| [`19_Regularized_Model_Validation_Loss_Comparison.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/19_Regularized_Model_Validation_Loss_Comparison.png) | Phase 3 (Deep Learning) | Loss stabilization curves demonstrating generalization gap elimination via BatchNorm/Dropout. |
| [`20_Tuned_Model_EarlyStopping_Loss_Convergence.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/20_Tuned_Model_EarlyStopping_Loss_Convergence.png) | Phase 3 (Deep Learning) | Optimal checkpoint convergence trajectory under `EarlyStopping` callback control. |
| [`21_Comparative_Training_History_All_NN.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/21_Comparative_Training_History_All_NN.png) | Phase 3 (Deep Learning) | 4-panel diagnostic matrix comparing training/val losses and accuracies across all 3 NN models. |
| [`22_Clinical_Confusion_Matrices_Evolution_NN.png`](file:///c:/Users/Amr%20Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Outputs/22_Clinical_Confusion_Matrices_Evolution_NN.png) | Phase 3 (Deep Learning) | 3-panel test set confusion matrix progression: Unregularized $\to$ Regularized $\to$ Tuned NN. |

---

## Project Directory Structure

```text
cardiac_monitoring_project/
├── Data/
│   ├── raw/                                        # Raw clinical dataset
│   │   └── heart.csv                               # Original patient data (918 records)
│   ├── processed/                                  # Processed caches & feature matrices
│   └── README.md                                   # Comprehensive data dictionary & clinical context
├── Notebooks/
│   ├── Baseline_Model.ipynb                        # Initial baseline exploration & prototype
│   ├── Cardiac_Patient_Monitoring_System.ipynb      # Main end-to-end capstone notebook (Phases 1-3)
│   └── best_cardiac_nn_model.keras                 # Checkpointed production Tuned Deep Neural Network
├── Outputs/                                        # 22 Publication-grade diagnostic visualization plots
│   ├── 01_Univariate_Categorical_Distributions.png
│   ├── 02_Univariate_Numerical_Distributions.png
│   ├── 03_Bivariate_Categorical_Features_vs_Target.png
│   ├── 04_Bivariate_Numerical_Features_Boxplots_vs_Target.png
│   ├── 05_Correlation_Matrix_and_Heatmap.png
│   ├── 06_Multivariate_Feature_Screening_Pairplot.png
│   ├── 07_Confusion_Matrix_and_ROC_Baseline_Pipeline.png
│   ├── 08_Elbow_Method_and_Silhouette_Scores_KMeans.png
│   ├── 09_Patient_Subgroups_KMeans_Clusters.png
│   ├── 10_DBSCAN_kNN_Distance_and_Clusters.png
│   ├── 11_Hierarchical_Clustering_Dendrogram.png
│   ├── 12_PCA_Scree_Plot_and_Cumulative_Variance.png
│   ├── 13_PCA_Loadings_Matrix_Heatmap.png
│   ├── 14_2D_PCA_Clinical_Projection.png
│   ├── 15_Manifold_Comparison_PCA_vs_tSNE.png
│   ├── 16_Activation_Functions_and_Derivatives.png
│   ├── 17_Learning_Rate_Dynamics_SGD_Convergence.png
│   ├── 18_Unregularized_Model_Training_vs_Validation_Loss.png
│   ├── 19_Regularized_Model_Validation_Loss_Comparison.png
│   ├── 20_Tuned_Model_EarlyStopping_Loss_Convergence.png
│   ├── 21_Comparative_Training_History_All_NN.png
│   └── 22_Clinical_Confusion_Matrices_Evolution_NN.png
├── requirements.txt                                # Locked environment dependencies
└── README.md                                       # Complete project documentation and benchmark report
```

---

## Setup & Reproduction Guide

### Prerequisites
- Python 3.10 or higher
- PowerShell (Windows) or Bash (Linux / macOS)
- NVIDIA GPU with CUDA support (optional, CPU execution supported)

### 1. Activate Virtual Environment
From the workspace root directory:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Project Dependencies
Ensure all libraries (Scikit-Learn, TensorFlow, Seaborn, Matplotlib) are installed:
```powershell
pip install -r requirements.txt
```

### 3. Launch the Jupyter Notebook
Open and run all cells sequentially in the primary notebook:
```powershell
jupyter notebook Notebooks/Cardiac_Patient_Monitoring_System.ipynb
```

### 4. Programmatic Model Inference
To load and run inference with the checkpointed production neural network:
```python
import tensorflow as tf
import numpy as np

# Load the checkpointed production model
model = tf.keras.models.load_model("Notebooks/best_cardiac_nn_model.keras")

# Generate predicted cardiac risk probabilities on preprocessed feature vectors
risk_probabilities = model.predict(X_test_proc)
binary_diagnoses = (risk_probabilities >= 0.50).astype(int)
```

---

## Authors & Institutional Attribution

- **Developer / Researcher:** Amr Sheqwara
- **Program:** AI & ML Internship Program, BinX Tech
- **Focus Area:** Supervised Classification, Unsupervised Phenotyping & Deep Neural Network Diagnostic Engineering