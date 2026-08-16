
<h1  style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">

  

AI & ML Course with BinX — Week 5: Unsupervised Learning & Capstone Project Kickoff

  

</h1>

  

<blockquote  style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

  

<b>Phase 2 to 3 Transition: Learning from Data That Has No Labels</b>

  

Welcome to Week 5 of the BinX Tech AI & ML Internship Program. This week marks our transition from supervised learning to unsupervised learning on unlabeled data. Interns cluster data with K-Means and DBSCAN, reduce high-dimensional data with PCA and t-SNE, detect anomalies with Isolation Forest, and finalize Phase 3 capstone project selection and Sprint 1 planning.

  

</blockquote>

  

---

  

## <span style="color:#F78BA0">Week Overview</span>

  

| Attribute | Details |
| :--- | :--- |
| **Curriculum Phase** | Phase 2 to Phase 3 Transition (40 hrs) |
| **Week Focus** | Unsupervised Learning, K-Means Clustering, Elbow Method, Silhouette Score, DBSCAN, Hierarchical Clustering, PCA, t-SNE, Anomaly Detection (Isolation Forest), Capstone Project Kickoff |
| **Primary Project / Dataset** | Heart Disease Dataset (`../../Projects/cardiac_monitoring_project/Data/heart.csv`) |
| **Deliverables** | Daily Jupyter Notebooks (`Task.ipynb`) & Daily Documentation (`README.md`) 
| **Tech Stack** | Python 3.10+, NumPy, Pandas, Matplotlib, Scikit-Learn, SciPy |

  

---

  

## <span style="color:#85C1E9">Weekly Learning Objectives</span>

  

-  **Unsupervised Learning Foundations:** Explain unsupervised learning and how it differs from supervised learning in terms of data, objectives, and evaluation.

  

-  **K-Means Clustering & Model Selection:** Partition unlabeled data into clusters and select optimal $k$ using the Elbow Method and Silhouette Score analysis.

  

-  **Density-Based & Hierarchical Clustering:** Apply DBSCAN for arbitrary cluster shapes and noise detection; construct and interpret hierarchical clustering dendrograms.

  

-  **Dimensionality Reduction:** Compress high-dimensional feature spaces with Principal Component Analysis (PCA), analyze explained variance ratios, and visualize local structures using t-SNE.

  

-  **Anomaly Detection:** Identify outliers and atypical observations in unlabeled data using Isolation Forest and density-based noise flagging.

  

-  **Capstone Kickoff & Sprint 1 Planning:** Define project problem statements, establish the Definition of Done, and draft Sprint 1 backlogs with concrete acceptance criteria.

---

## <span style="color:#309c42ff">Week 5 Deliverables & Technical Discipline</span>

  

1.  **Feature Scaling Discipline:** Unsupervised algorithms (such as K-Means and PCA) rely on Euclidean distances and variance maximization. Features must always be scaled with `StandardScaler` to prevent high-magnitude features from dominating calculations.

2.  **Quantitative Validation:** Since unsupervised learning lacks ground-truth labels, cluster quality and dimensionality selections must be justified using internal metrics (`silhouette_score`, `inertia_`, `explained_variance_ratio_`).

3.  **Domain Interpretation:** Cluster centroids and principal components must be mapped back to original feature spaces to extract actionable clinical and business insights.

4.  **Capstone Readiness:** Day 5 bridges Phase 2 and Phase 3. No capstone implementation begins without mentor approval of the project problem statement and Sprint 1 scope.