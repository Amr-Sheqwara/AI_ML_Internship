
<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 5 — Day 4: t-SNE Visualization & Anomaly Detection
</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 4 Focus: Non-Linear Manifold Projection, Local Neighborhood Preservation & Tree-Based Anomaly Isolation</b>

Today we apply <b>t-Distributed Stochastic Neighbor Embedding (t-SNE)</b> to visualize high-dimensional non-linear cluster structures and distinguish its local-neighborhood preservation from the global-variance preservation of <b>Principal Component Analysis (PCA)</b>. Furthermore, we explore unsupervised <b>Anomaly Detection</b> using <b>Isolation Forest</b>, quantify flagged outlier instances based on recursive random partitioning, and clinically evaluate why extreme observations are isolated in early tree splits.

</blockquote>

---

## <span style="color:#F78BA0">4.1 Overview & Objectives</span>

<blockquote style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

In unsupervised learning, complex datasets frequently exhibit non-linear manifold geometries that linear orthogonal projections (such as PCA) cannot fully unpack without overlap. In parallel, real-world datasets contain rare, atypical observations (anomalies, fraud, defects, or extreme pathological lesions) that sit far from dense data masses. Day 4 explores local neighbor embeddings for exploratory visualization and tree-based partition algorithms to isolate anomalies without ground-truth labels.

<b>Key Objectives:</b>

- **t-SNE for Manifold Visualization:** Compress 30 continuous clinical dimensions to 2D using Student-t kernel similarity mappings while avoiding the crowding problem.
- **PCA vs. t-SNE Comparative Evaluation:** Contrast linear global-variance retention against non-linear local neighborhood preservation across axes interpretability, computational complexity, and modeling utility.
- **Scaling Discipline:** Enforce zero mean ($\mu = 0$) and unit variance ($\sigma = 1$) standardization via `StandardScaler` prior to manifold embedding and isolation trees.
- **Anomaly Detection Principles:** Explain why anomaly detection is fundamentally unsupervised and how outliers sit apart from dense clusters.
- **Isolation Forest Mechanics:** Implement random recursive partitioning and tune the `contamination` threshold ($0.05$) to measure anomaly scores via average tree path length.
- **Clinical Anomaly Inspection:** Quantitatively inspect flagged anomalous points using standard deviation deviations ($Z$-scores) to construct data-driven isolation hypotheses.

</blockquote>

---

## <span style="color:#85C1E9">4.2 Dataset Information</span>

We utilize the **Breast Cancer Wisconsin Diagnostic Dataset** (`Breast_Cancer_Wisconsin.csv`) for high-dimensional manifold projection and clinical outlier detection.

- **Source File:** `../../Data/Breast_Cancer_Wisconsin.csv`
- **Target Variable ($y$):** `diagnosis` (`M` = Malignant: 212, `B` = Benign: 357)
- **Feature Matrix ($X$):** 30 continuous nuclear morphological measurements (`radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, and `fractal dimension` across `mean`, `se`, and `worst`).
- **Instances:** 569 complete clinical records (zero missing values after dropping `id` and `Unnamed: 32`).
- **Preprocessing:** All 30 continuous features standardized to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$) via `StandardScaler`.

---

## <span style="color:#F8C471">4.3 Key Concepts & Workflow</span>

### 1. t-SNE for Local-Structure Visualization
t-SNE converts Euclidean distances between data points into conditional probabilities that represent similarities:
- **High-Dimensional Space:** Uses a Gaussian probability distribution to model pairwise neighbor similarities.
- **Low-Dimensional (2D) Space:** Uses a heavy-tailed Student-$t$ distribution (1 degree of freedom) to resolve the crowding problem.
- **Optimization:** Minimizes the Kullback-Leibler (KL) divergence between the high-dimensional and low-dimensional probability distributions via gradient descent.
- **Perplexity:** Controls the effective number of local nearest neighbors considered by each point (standard value: $30$).

---

### 2. Method Comparison: PCA vs. t-SNE

| Attribute | Principal Component Analysis (PCA) — Day 3 | t-Distributed Stochastic Neighbor Embedding (t-SNE) — Day 4 |
| :--- | :--- | :--- |
| **Mathematical Approach** | Linear orthogonal transformation (Eigenvalues / SVD). | Non-linear probabilistic manifold neighbor embedding (KL divergence). |
| **Preserves** | **Global Variance:** Captures maximum dataset variance. | **Local Neighborhoods:** Preserves local cluster proximity. |
| **Main Use Case** | Feature compression, noise reduction, and model input. | Exploratory data analysis and non-linear cluster visualization. |
| **Speed** | Fast ($O(\min(n^3, d^3))$); scalable to large datasets. | Slower ($O(n^2)$ / $O(n \log n)$ via Barnes-Hut). |
| **Axes Meaning** | Interpretable linear eigenvectors with variance ratios. | **No physical meaning:** Coordinates are arbitrary relative positions. |
| **Downstream Modeling** | Transforms new out-of-sample test records directly. | **Visualization only:** Cannot project new unseen records. |

---

### 3. Isolation Forest for Anomaly Detection
Anomalies are rare observations that deviate significantly from the normal data distribution:
- **Partitioning Principle:** Isolation Forest randomly selects a feature and a random split value between that feature's minimum and maximum.
- **Short Path Length:** Because anomalous points sit in sparse regions far from the dense core mass, they are separated into solitary leaf nodes with very few partition splits (short tree depth $h(x)$). Normal points in dense clusters require deep recursive splitting.
- **Contamination:** Specifies the expected proportion of outliers (set to `contamination=0.05`, flagging the top $5\%$ most isolated records).
- **DBSCAN Overlap:** Connects directly to Day 2 DBSCAN noise points (`label == -1`), confirming that density-based sparsity identifies structural outliers.

---

### 4. Anomaly Inspection Summary & Hypotheses

Out of 569 observations, Isolation Forest flagged **29 clinical anomalies** ($5.10\%$):

| Flagged Record | Ground Truth | Anomaly Score | Key Clinical Metric Deviations | Isolation Hypothesis |
| :--- | :---: | :---: | :--- | :--- |
| **Index 0** | Malignant (`M`) | **-0.1256** | `concavity_mean = 0.3001` ($Z = +2.65$)<br>`area_worst = 2019.0` ($Z = +2.48$) | **Severe Macro-Irregularity:** Combines massive nuclear volume with extreme boundary concavity (top 1%), resulting in immediate isolation at root tree levels. |
| **Index 122** | Malignant (`M`) | **-0.1084** | `radius_mean = 24.25` ($Z = +2.87$)<br>`area_mean = 1878.0` ($Z = +3.46$)<br>`area_worst = 3216.0` ($Z = +4.58$) | **Giant Nuclear Outlier:** Extreme dimensional outlier across all spatial axes (>3.4 standard deviations above mean), isolated in 1 or 2 random cuts. |


