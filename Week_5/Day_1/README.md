<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 5 — Day 1: Unsupervised Learning & K-Means Clustering
</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 1 Focus: Discovering Hidden Structure, Scaling Discipline & Optimizing k</b>

Today we transition from supervised learning to unsupervised learning on unlabeled data. We apply <b>K-Means Clustering</b> to discover natural groupings without pre-defined targets, use <b>StandardScaler</b> feature scaling to prevent distance distortion, optimize the number of clusters $k$ using the <b>Elbow Method</b> and <b>Silhouette Score</b>, and interpret the clinical characteristics of each discovered patient cluster.

</blockquote>

---

## <span style="color:#F78BA0">1.1 Overview & Objectives</span>

<blockquote style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

In supervised learning (Weeks 3-4), every training example had a known correct answer (the target $y$). Unsupervised learning works on data with no labels at all — there is no $y$. Instead of predicting a known answer, the goal is to discover hidden structure the data contains on its own: natural groupings, underlying dimensions, or unusual points.

<b>Key Objectives:</b>

- Explain unsupervised learning and how it differs from supervised learning in data structure, goals, and evaluation.
- Clean and extract numeric clinical features from the Heart Disease dataset (`heart.csv`).
- Apply `StandardScaler` feature scaling prior to clustering to prevent distance dominance.
- Execute K-Means clustering across $k \in [1, 10]$ and locate the inflection point using the **Elbow Method** (inertia).
- Quantitatively validate candidate $k$ values using the **Silhouette Score**.
- Fit the final K-Means model, extract cluster centroids, and profile the clinical characteristics of each patient group in Markdown.

</blockquote>

---

## <span style="color:#85C1E9">1.2 Dataset Information</span>

We utilize the **Heart Disease Clinical Dataset** from the Cardiac Monitoring Project for patient segmentation and pattern discovery.

- **Source File:** `../../Projects/cardiac_monitoring_project/Data/heart.csv`
- **Target Handling:** The target label (`target`) is dropped during clustering to evaluate pure unsupervised partitioning.
- **Key Numeric Features ($X$):**
  - `age`: Patient age in years
  - `trestbps`: Resting blood pressure (in mm Hg on admission to the hospital)
  - `chol`: Serum cholesterol in mg/dl
  - `thalach`: Maximum heart rate achieved during exercise
  - `oldpeak`: ST depression induced by exercise relative to rest
- **Instances:** 303 clinical records (clean, zero-missing rows)

---

## <span style="color:#F8C471">1.3 Key Concepts & Workflow</span>

### 1. Supervised vs. Unsupervised Learning

| Dimension | Supervised Learning | Unsupervised Learning |
| :--- | :--- | :--- |
| **Data** | Has labels ($X$ and $y$) | No labels ($X$ only) |
| **Goal** | Predict the known target | Discover hidden structure |
| **Examples** | Regression, classification | Clustering, dimensionality reduction, anomaly detection |
| **Evaluation** | Compare prediction to true label | No ground truth — uses internal metrics and judgment |

### 2. K-Means Centroid-Assignment Algorithm
K-Means partitions data into a chosen number of clusters ($k$) by repeating two steps until stable:
- **Step 1:** Place $k$ cluster centers ("centroids"), initially at random.
- **Step 2:** Assign each point to its nearest centroid.
- **Step 3:** Move each centroid to the mean position of the points assigned to it.
- **Repeat Steps 2-3:** Iterate until the centroids stop moving.

### 3. Scaling Before Clustering
Always scale features before clustering. K-Means uses distance, so an unscaled large-range feature (such as `chol` spanning 126–564) would dominate a small-range one (such as `oldpeak` spanning 0.0–6.2).

### 4. Choosing $k$: The Elbow Method & Silhouette Score
- **The Elbow Method:** Runs K-Means for a range of $k$ values ($k \in [1, 10]$) and plots the inertia (total within-cluster distance) against $k$. The point where the rate of improvement drops sharply forms an "elbow", indicating the recommended $k$.
- **The Silhouette Score:** Measures how well each point sits inside its own cluster versus the nearest other cluster, on a scale from $-1$ to $+1$. A higher average silhouette score confirms better-defined clusters.

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1) Load and scale continuous clinical features
df = pd.read_csv('../../Projects/cardiac_monitoring_project/Data/heart.csv')
features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X_scaled = StandardScaler().fit_transform(df[features])

# 2) Elbow Method (Inertia vs. k)
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in range(1, 11)]

# 3) Silhouette Score Validation
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    print(f"k={k} -> Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")

# 4) Fit Final Model
final_km = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = final_km.fit_predict(X_scaled)
```

---

## <span style="color:#309c42ff">1.4 Repository Structure</span>

- `Task.ipynb`: Jupyter Notebook implementing data loading, feature scaling with `StandardScaler`, inertia calculation across $k \in [1, 10]$, silhouette score validation, final K-Means model fitting, 2D centroid visualization, and patient profile interpretation.
- `README.md`: Day 1 technical documentation detailing unsupervised theory, mathematical formulations, elbow analysis, and clinical cluster interpretation.