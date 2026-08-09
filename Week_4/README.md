
<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 4: Evaluation, Tuning & Pipelines

</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

🎯 <b>Phase 2 Milestone: Turning Models That Run into Models That Are Trustworthy</b>

Welcome to Week 4 of the BinX Tech AI & ML Internship Program. This week focuses on robust model evaluation, diagnosing overfitting/underfitting via the bias-variance trade-off, systematic hyperparameter optimization with `GridSearchCV`, and packaging end-to-end workflows into clean, leakage-free <b>Scikit-Learn Pipelines</b>.

</blockquote>

---

## <span style="color:#F78BA0">Week Overview</span>

| Attribute | Details |
| :--- | :--- |
| **Curriculum Phase** | Phase 2: Core ML Training (40 hrs) |
| **Week Focus** | Three-Way Data Splitting, $k$-Fold Cross-Validation, Bias-Variance Trade-Off, Feature Engineering, GridSearchCV, Scikit-Learn Pipelines |
| **Primary Dataset** | Palmer Archipelago Penguins Dataset (`penguins_size.csv`) |
| **Deliverables** | Daily Jupyter Notebooks (`Task.ipynb`) & Daily Documentation (`README.md`) |
| **Tech Stack** | Python 3.10+, NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn |

---

## <span style="color:#85C1E9">Weekly Learning Objectives</span>

- ✂️ **Three-Way Data Splitting:** Master partitioning datasets into Train (60%), Validation (20%), and Test (20%) sets to eliminate test information leakage.

- 🔄 **Cross-Validation:** Evaluate classification and regression models reliably with 5-fold and Stratified $k$-Fold cross-validation (`cross_val_score`).

- ⚖️ **Bias-Variance Trade-Off:** Diagnose underfitting vs. overfitting from train-vs-validation score gaps and apply regularization (Ridge $L_2$, Lasso $L_1$).

- ⚙️ **Feature Engineering & Tuning:** Create domain-informed features, encode/scale inputs properly, and automate search using `GridSearchCV`.

- 📦 **Scikit-Learn Pipelines:** Build end-to-end, leakage-free workflows combining `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, and estimators into unified `Pipeline` objects.

---
