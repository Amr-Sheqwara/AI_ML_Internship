
<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 5 — Day 3: Dimensionality Reduction with PCA
</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 3 Focus: Overcoming the Curse of Dimensionality, Scaling Discipline & Explained Variance Optimization</b>

Today we apply <b>Principal Component Analysis (PCA)</b> to compress high-dimensional feature spaces into orthogonal axes of maximum variance. We enforce <b>StandardScaler</b> discipline to prevent scale dominance, analyze individual and cumulative <b>Explained Variance Ratios</b> to select an optimal component count retaining $\approx 95\%$ of information, project data into a 2D subspace for visual class inspection, and evaluate the trade-offs between information retention and feature interpretability.

</blockquote>

---

## <span style="color:#F78BA0">3.1 Overview & Objectives</span>

<blockquote style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

Real-world datasets often have dozens or hundreds of features. High dimensionality causes data to become sparse, distances to lose discriminative meaning, models to overfit more easily, and prevents visualization beyond three dimensions. Dimensionality reduction compresses many features into a few, keeping as much information as possible while making data easier to model and visualize.

<b>Key Objectives:</b>

-  **The Curse of Dimensionality:** Explain why high-dimensional feature spaces degrade distance-based estimators and cause overfitting.
-  **Scaling Discipline:** Standardize all continuous clinical features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$) with `StandardScaler` prior to PCA.
-  **Full PCA & Cumulative Scree Analysis:** Fit a 30-component PCA model and construct the cumulative explained variance curve.
-  **Variance Optimization:** Quantitatively determine the minimum component count required to retain $\ge 95\%$ of total dataset variance.
-  **2D Diagnostic Projection:** Project 30 dimensions down to 2 principal components to visually separate Malignant and Benign tumors.
-  **Trade-Off Assessment:** Document what the reduction preserved (global variance, class separation, orthogonality) and what it cost (loss of direct physical feature interpretability).

</blockquote>

---

## <span style="color:#85C1E9">3.2 Dataset Information</span>

We utilize the **Breast Cancer Wisconsin Diagnostic Dataset** (`Breast_Cancer_Wisconsin.csv`) for high-dimensional feature compression and orthogonal subspace projection.

- **Source File:** `../../Data/raw/Breast_Cancer_Wisconsin.csv`
- **Target Variable ($y$):** `diagnosis` (`M` = Malignant: 212, `B` = Benign: 357)
- **Feature Matrix ($X$):** 30 continuous morphological measurements of cell nuclei (`radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, and `fractal dimension` across `mean`, `se`, and `worst`).
- **Instances:** 569 complete clinical records (zero missing values after dropping `id` and `Unnamed: 32`).

---

## <span style="color:#F8C471">3.3 Key Concepts & Workflow</span>

### 1. The Curse of Dimensionality
- **Spatial Sparsity:** As dimensions increase, data points become exponentially sparse, reducing model generalizability.
- **Distance Distortion:** In high-dimensional spaces, pairwise Euclidean distances converge, diminishing cluster and separation contrast.

### 2. Principal Component Analysis (PCA)
- **Orthogonal Variance Maximization:** Finds new orthogonal axes that maximize the variance of projected points.
- **Scaling Requirement:** Features must always be standardized via `StandardScaler` prior to PCA, because PCA is variance-dependent and unscaled large-magnitude features would artificially dominate the principal directions.

### 3. Cumulative Variance Breakdown (First 10 Components)

| Principal Component | Individual Explained Variance | Cumulative Explained Variance |
| :--- | :---: | :---: |
| **PC1** | 44.27% | 44.27% |
| **PC2** | 18.97% | 63.24% |
| **PC3** | 9.39% | 72.64% |
| **PC4** | 6.60% | 79.24% |
| **PC5** | 5.50% | 84.73% |
| **PC6** | 4.02% | 88.76% |
| **PC7** | 2.25% | 91.01% |
| **PC8** | 1.59% | 92.60% |
| **PC9** | 1.39% | 93.99% |
| **PC10** | 1.17% | **95.16%** |