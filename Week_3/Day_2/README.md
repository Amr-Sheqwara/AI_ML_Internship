
<h1 style="color:#9f43c3ff; border-bottom: 3px solid #9f43c3ff; padding-bottom: 8px;">
AI & ML Course with BinX — Week 3 — Day 2: Linear Regression & Model Evaluation
</h1>

<blockquote  style="border-left: 3px solid #2e1a9aff; padding-left: 12px; margin-left: 0;">
📈 <b>Day 2 Focus: Linear Regression Fitting, Feature Interpretation & Baseline Benchmarking</b>
  
Today we trained our first predictive regression model using <b>Scikit-Learn</b> (`LinearRegression`). We performed data cleaning, split features ($X$) and target ($y$), extracted feature coefficients alongside the bias intercept, evaluated performance using standard regression metrics (MAE, RMSE, $R^2$), and benchmarked against a naive mean baseline.
</blockquote>

---
## <span style="color:#309c42ff">2.1 Overview & Learning Objectives</span>
  
<blockquote  style="border-left: 3px solid #FD1D1D; padding-left: 12px; margin-left: 0;">
<b>Key Objectives:</b>

- 📂 Load house price dataset (`HousePriceDataset.csv`) and clean non-numeric attributes (`street`, `city`, `statezip`, `country`, `date`).

- 🎯 Define Feature Matrix ($X$) and Target Vector ($y$).

- 🤖 Train a <code>LinearRegression</code> model on an 80/20 train-test split (`random_state=42`).

- 📊 Extract feature coefficients (`model.coef_`) and intercept (`model.intercept_`) to identify the strongest predictor.

- 📐 Evaluate model performance using MAE, RMSE, and $R^2$.

- 🎯 Benchmark model RMSE against a naive mean baseline (`y_train.mean()`) to verify value addition.
</blockquote>

---

## <span style="color:#309c42ff">2.2 Dataset Information</span>

We utilize the **House Price Dataset** containing property characteristics and historical sales values.

-  <b>Source File:</b> `../../Data/raw/HousePriceDataset.csv`

-  <b>Target Variable ($y$):</b> `price` (Continuous target value)

-  <b>Features ($X$):</b> `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`

---

## <span style="color:#309c42ff">2.3 Key Results & Summary</span>

### 1. Model Intercept & Strongest Predictors

-  <b>Model Intercept (Bias):</b> Evaluated during model fitting.

-  <b>Top Positive Predictor (Categorical/Binary):</b> `waterfront` (+382,459.67 per unit)

-  <b>Key Numerical Predictors:</b> `floors` (+69,824.74), `view` (+44,755.84), `bathrooms` (+36,520.44), and `sqft_living` (+186.05 per sqft).

### 2. Performance Metrics vs. Baseline

| Metric | Model Score | Baseline (Mean Predictor) | Improvement |
|---|---|---|---|
| **MAE** | $210,908.17 | — | — |
| **RMSE** | $993,439.36 | $1,010,500.78 | **-$17,061.41** (Value Added ✅) |
| **$R^2$** | 0.0323 (3.23%) | 0.00 | +0.0323 |

---

## <span style="color:#309c42ff">2.4 Repository Contents</span>

- 📓 [Task.ipynb]: Jupyter Notebook implementing linear regression, coefficient extraction, metric computation, and baseline evaluation.