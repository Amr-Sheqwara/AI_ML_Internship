
<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 5 — Day 2: DBSCAN, Hierarchical Clustering & Model Comparison
</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 2 Focus: Overcoming K-Means Limitations with Density-Based & Agglomerative Hierarchical Clustering</b>

Today we explore alternative clustering algorithms to address key weaknesses in K-Means. We apply <b>DBSCAN (Density-Based Spatial Clustering)</b> to automatically discover clusters of arbitrary shapes and identify outlier noise points, construct <b>Hierarchical Clustering Dendrograms</b> using Ward's minimum variance linkage, and evaluate side-by-side performance across K-Means, DBSCAN, and Hierarchical clustering on the Palmer Archipelago Penguins dataset.

</blockquote>

---

## <span style="color:#F78BA0">2.1 Overview & Objectives</span>

<blockquote style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

K-Means has real limitations: it requires choosing $k$ in advance, assumes clusters are round and similarly sized, and forces every point into a cluster — even clear outliers. When data has irregular shapes or noise, K-Means gives misleading results. Day 2 introduces two alternative paradigms: density-based spatial partitioning and hierarchical tree-based clustering.

<b>Key Learning Objectives:</b>

- Explain K-Means limitations and determine when alternative clustering methods are preferable.
- Run DBSCAN, tune neighborhood radius (`eps`) and minimum samples (`min_samples`), and interpret discovered clusters and noise points (`label == -1`).
- Build and interpret an agglomerative hierarchical clustering dendrogram and select an optimal cut height.
- Compare K-Means, DBSCAN, and Hierarchical clustering side-by-side using 2D scatter visualizations and Silhouette Scores.
- Document and justify which clustering technique best fits the morphological geometry of the dataset.

</blockquote>

---

## <span style="color:#85C1E9">2.2 Dataset Information</span>

We utilize the **Palmer Archipelago Penguins Morphological Dataset** (`penguins_size.csv`) for unsupervised biometric clustering and species segmentation.

- **Source File:** `../../Data/penguins_size.csv`
- **Target & Identifier Handling:** Categorical columns (`species`, `island`, `sex`) are excluded from clustering to evaluate pure unsupervised grouping.
- **Continuous Morphological Features ($X$):**
  - `culmen_length_mm`: Culmen length in millimeters
  - `culmen_depth_mm`: Culmen depth in millimeters
  - `flipper_length_mm`: Flipper length in millimeters
  - `body_mass_g`: Body mass in grams
- **Preprocessing:** Records with missing values were removed, resulting in 334 complete morphological observations standardized to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$) via `StandardScaler`.

---

## <span style="color:#F8C471">2.3 Key Concepts & Workflow</span>

### 1. Why K-Means Isn't Always Enough
- **Spherical Cluster Assumption:** Assumes clusters are convex, isotropic, and similarly sized.
- **Rigid Outlier Assignment:** Forces every point into a centroid, distorting cluster boundaries when noise is present.
- **Mandatory Hyperparameter $k$:** Requires specifying the exact number of clusters beforehand.

### 2. DBSCAN (Density-Based Spatial Clustering)
DBSCAN groups points that are packed closely together in dense regions and marks points in sparse regions as noise:
- **`eps`:** Maximum distance between two samples for one to be considered in the neighborhood of the other.
- **`min_samples`:** Number of samples in a neighborhood for a point to qualify as a core point.
- **Outlier Flagging:** Assigns label `-1` to noise points, preventing distortion of valid clusters.

### 3. Hierarchical Clustering & Dendrograms
Hierarchical clustering builds a tree of nested clusters starting with every point as its own cluster and iteratively merging the closest pairs:
- **Linkage Criterion (Ward's Method):** Minimizes the total within-cluster variance.
- **Dendrogram:** A tree diagram representing merge distances, allowing a cut at any distance threshold to obtain the desired cluster partition without specifying $k$ in advance.

### 4. Method Comparison Matrix

| Method | Best When | Watch Out For |
| :--- | :--- | :--- |
| **K-Means** | Round, similarly-sized clusters; $k$ roughly known | Forces outliers into clusters; requires pre-specifying $k$ |
| **DBSCAN** | Irregular shapes; noise/outliers present in data | Sensitive to `eps`; struggles with varying cluster densities |
| **Hierarchical** | Nested structure or dendrogram visualization desired | Computationally expensive on large datasets |

