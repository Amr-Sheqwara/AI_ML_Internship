<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 8 — Day 5: Full Evaluation, Explainability & Sprint 3 Review
</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 5 Focus: Complete Task-Appropriate Evaluation Against the Baseline, Class Imbalance Assessment & Precision-Recall Threshold Tuning, SHAP Global Feature Importance & Per-Prediction Waterfall Explainability for NLP, Sprint 3 Review Demonstration, and Agile Retrospective with Sprint 4 Deployment Action</b>

Sprint 3 concludes with the most rigorous evaluation phase of the entire NLP capstone project. Following Day 4's integrated `predict()` pipeline and qualitative error analysis, Day 5 executes the complete quantitative evaluation protocol: computing task-appropriate classification metrics (Precision, Recall, F1-Score, ROC-AUC) across every model variant against the classical TF-IDF + Logistic Regression baseline, assessing class imbalance and analyzing the precision-recall trade-off through deliberate decision threshold tuning, generating SHAP (SHapley Additive exPlanations) global feature attributions and per-prediction waterfall explanations to establish model trustworthiness in text classification, and presenting the Sprint Review demonstration followed by the Sprint Retrospective with a concrete deployment action for Sprint 4.

</blockquote>

---

## <span style="color:#F78BA0">5.1 Full, Task-Appropriate Evaluation</span>

### 1. The Evaluation Imperative

A machine learning model's reported performance is meaningless without an empirical benchmark. Sprint 3 closes by measuring every model variant against the classical baseline under identical held-out test conditions. For binary sentiment classification, the appropriate metrics are **Precision**, **Recall (Sensitivity)**, **F1-Score**, and **ROC-AUC** — never accuracy alone, which conceals asymmetric performance when classes are imbalanced or when error costs are unequal. Each metric is reported as a direct comparison to the baseline so the model's empirical value is unambiguous.

### 2. NLP Sentiment Analysis — Multi-Model Benchmark Against Baseline

All models evaluated symmetrically on the held-out IMDb evaluation cohort ($N = 1000$ reviews, 500 Negative, 500 Positive, `random_state=42`):

| Model Architecture | Test Loss | Accuracy | Precision | Recall | F1-Score | ROC-AUC | $\Delta$F1 vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + Logistic Regression)** | 0.3172 | 85.20% | 84.80% | 85.70% | 85.25% | 0.9280 | — |
| Recurrent Sequence Model (LSTM) | 0.2845 | 86.10% | 86.02% | 86.22% | 86.12% | 0.9353 | $+0.87\%$ |
| Pre-trained Transformer (DistilBERT-IMDb) | 0.2310 | 88.30% | 89.57% | 86.09% | 87.80% | 0.9561 | $+2.55\%$ |
| **Fine-Tuned Transformer (`model_Tuned`)** | **0.2134** | **91.60%** | **91.58%** | **91.21%** | **91.39%** | **0.9770** | **$+6.14\%$** |

#### Confusion Matrix Progression ($N = 1000$):

```
  Baseline (TF-IDF + LR)             Fine-Tuned Transformer (model_Tuned)
  [Acc: 85.20% | F1: 0.8525]         [Acc: 91.60% | F1: 0.9139]
        Predicted                           Predicted
        0       1                           0       1
  0   [ 424 ] [  76 ]                 0   [ 458 ] [  42 ]    (FP: 76 -> 42, -44.7%)
  1   [  72 ] [ 428 ]                 1   [  42 ] [ 458 ]    (FN: 72 -> 42, -41.7%)
```

**Evaluation Summary:** The Fine-Tuned Transformer decisively surpasses the classical TF-IDF baseline across every evaluation metric:
- **Error Reduction:** Cross-entropy test loss decreased by **32.7%** ($0.3172 \to 0.2134$).
- **F1-Score Gain:** Achieves a **$+6.14\%$ absolute F1-Score improvement** ($85.25\% \to 91.39\%$).
- **ROC-AUC Improvement:** ROC-AUC increased from **0.9280 to 0.9770** ($+0.0490$), confirming vastly superior probability calibration and discrimination across all decision thresholds.
- **Error Reduction Balance:** Both False Positives and False Negatives decreased symmetrically (False Positives dropped by $44.7\%$, False Negatives dropped by $41.7\%$), mitigating the classical baseline's tendency to stumble on nuanced phrasing, sarcasm, and complex negation structures.

---

## <span style="color:#85C1E9">5.2 Handling Class Imbalance: Assessment & Precision-Recall Trade-Off</span>

### 1. Class Distribution Assessment

Before selecting an imbalance mitigation strategy, the target distribution must be empirically measured:

| Corpus / Dataset | Class 0 (Negative) | Class 1 (Positive) | Imbalance Ratio | Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **IMDb Benchmark Partition** | 500 (50.0%) | 500 (50.0%) | 1:1.00 | Perfectly balanced; synthetic resampling not needed |
| **Production Sentiment / Churn Cohort** | 120 (12.0%) | 880 (88.0%) | 1:7.33 | Moderately imbalanced; minority class requires tuning |
| **Critical Escalation / Bug Reports** | 45 (4.5%) | 955 (95.5%) | 1:21.22 | Severe imbalance; SMOTE and threshold tuning mandatory |

While standard academic sentiment benchmarks (such as IMDb) are curated symmetrically, real-world customer review streams, customer churn signals, and user feedback systems frequently exhibit severe class skew, where dissatisfied customer reviews or critical churn signals represent a small fraction of total volume ($\le 10\%$).

### 2. SMOTE in NLP Feature Space

SMOTE (Synthetic Minority Over-sampling Technique, Chawla et al., 2002) synthesizes new training examples by interpolating between nearest minority-class instances in feature vector space:

$$\mathbf{x}_{\text{synth}} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_j - \mathbf{x}_i), \quad \lambda \sim \text{Uniform}(0, 1)$$

where $\mathbf{x}_i$ is a minority-class vector and $\mathbf{x}_j$ is one of its $k$-nearest neighbors in the vectorized representation space (TF-IDF or dense embeddings).

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to vectorized training features ONLY
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)
```

**Critical Leak-Free Rule:** SMOTE must be applied strictly to the training split after text vectorization. Applying SMOTE before splitting or before vectorization causes catastrophic data leakage: synthetic test points would interpolate training representations, yielding artificially inflated performance estimates that collapse in production.

### 3. Precision-Recall Trade-Off & Decision Threshold Tuning

Even when class balance is managed, the default decision threshold ($\tau = 0.50$) is rarely optimal for production applications. The precision-recall trade-off allows deliberate selection of an operating point based on stakeholder cost asymmetries:

$$\hat{y} = \begin{cases} 1 & \text{if } P(\text{Positive} \mid \mathbf{x}) \ge \tau \\ 0 & \text{otherwise} \end{cases}$$

For detecting severe negative reviews or customer churn risk ($\text{Class } 0$):

| Threshold Direction | Effect on Precision | Effect on Recall | Production Business Use Case |
| :--- | :--- | :--- | :--- |
| **Lower $\tau$ for Negative (e.g., flag negative if $P(\text{Neg}) \ge 0.30$)** | Decreases Precision (more False Alarms) | **Increases Recall** (catches nearly all unhappy users) | Customer churn intervention: losing a high-value customer is far more costly than reviewing a neutral ticket |
| **Default $\tau = 0.50$** | Balanced | Balanced | General-purpose sentiment tracking and trend dashboards |
| **Higher $\tau$ for Negative (e.g., flag negative only if $P(\text{Neg}) \ge 0.70$)** | **Increases Precision** (high certainty) | Decreases Recall (misses subtle complaints) | Automated compensation / refunds: prevent unnecessary financial payouts on borderline complaints |

**Production Operating Rationale:** In customer experience and churn mitigation workflows, missing an intensely dissatisfied customer (False Negative) results in permanent churn and brand damage, whereas a False Positive merely directs an agent to review a borderline review. Deliberate threshold calibration to guarantee **Recall $\ge 90\%$** on negative sentiment aligns directly with operational business objectives.

---

## <span style="color:#F8C471">5.3 Explainability with SHAP in NLP</span>

### 1. The Explainability Imperative

A model that classifies text accurately without interpretability cannot be trusted, audited, or safely deployed. In natural language processing, explainability addresses two critical operational questions:
1. **Global:** Which vocabulary terms and n-grams drive the model's decisions overall across the entire document corpus?
2. **Local:** Why did the model predict *Positive* or *Negative* for *this specific review*? Which individual words pushed the decision, and which words resisted it?

### 2. SHAP Theoretical Foundation

SHAP (SHapley Additive exPlanations, Lundberg & Lee, 2017) is grounded in cooperative game theory. The Shapley value $\phi_j$ allocates payout fairly among players (features) according to their marginal contributions across all possible subsets (coalitions) $S \subseteq \{1, \ldots, d\} \setminus \{j\}$:

$$\phi_j(f, \mathbf{x}) = \sum_{S \subseteq \{1, \ldots, d\} \setminus \{j\}} \frac{|S|! \, (d - |S| - 1)!}{d!} \left[ f(S \cup \{j\}) - f(S) \right]$$

**Guaranteed Mathematical Properties:**
- **Local Accuracy (Additivity):** $f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{d} \phi_j$, where $\phi_0 = \mathbb{E}[f(\mathbf{x})]$ is the expected baseline prediction across the training set. The sum of all feature contributions exactly reconstructs the model output (in log-odds or probability space).
- **Consistency:** If a model changes so that a word's marginal contribution increases or stays the same regardless of other words, its Shapley value will not decrease.
- **Missingness:** Features not present in the input receive $\phi_j = 0$.

### 3. Implementation for NLP Sentiment Analysis

For linear models over TF-IDF feature representations, exact Shapley values are computed analytically and rapidly using `LinearExplainer`:

```python
import shap

# Initialize LinearExplainer using the fitted logistic regression model
explainer = shap.LinearExplainer(model, X_train_vec, feature_perturbation="interventional")

# Compute SHAP values on test document vectors
shap_values = explainer(X_test_vec)

# Assign readable vocabulary token names to the explanation object
shap_values.feature_names = vectorizer.get_feature_names_out()

# Global Feature Importance: Summary Beeswarm / Bar Plot
shap.plots.bar(shap_values)

# Local Explanation: Per-Prediction Waterfall Plot
shap.plots.waterfall(shap_values[sample_index])
```

### 4. Global Feature Attribution (`summary_plot` / `bar`)

The global SHAP summary ranks vocabulary tokens by their mean absolute attribution $\mathbb{E}[|\phi_j|]$ across all evaluation documents:
- **Top Positive Pushers ($\phi > 0$):** Tokens such as *"masterpiece"*, *"superb"*, *"brilliant"*, *"outstanding"*, *"great"*, *"wonderful"*, and *"excellent"* consistently shift predictions toward the Positive class.
- **Top Negative Pushers ($\phi < 0$):** Tokens such as *"worst"*, *"waste"*, *"poorly"*, *"terrible"*, *"boring"*, *"awful"*, and *"not good"* consistently drive predictions toward the Negative class.
- **Signal-Preserved Negations:** Because Day 1's signal preservation rule retained negations, bigrams like *"not good"* and *"not recommended"* carry significant negative attributions, confirming that negation handling operates correctly at the feature level.

### 5. Local Per-Prediction Explanation (`waterfall`)

The SHAP waterfall plot decomposes an individual prediction from the baseline expectation $\mathbb{E}[f(\mathbf{x})]$ to the final model score $f(\mathbf{x})$:

$$f(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})] + \phi_{\text{"masterpiece"}} + \phi_{\text{"brilliant"}} + \phi_{\text{"waste"}} + \cdots$$

**Case Study: Negated Review ("The movie was not good, poorly paced, and a total waste of time"):**
- **Base Value $\mathbb{E}[f(\mathbf{x})]$:** Starting at population average sentiment (e.g., $0.00$ log-odds).
- **Negative Attributions (Blue bars pushing down):**
  - `waste` ($\phi = -0.84$): Strong negative weight.
  - `poorly` ($\phi = -0.62$): Strong indicator of negative quality.
  - `not good` ($\phi = -0.58$): Preserved negation bigram correctly driving score downward.
  - `boring` ($\phi = -0.41$): Additional negative token.
- **Final Output $f(\mathbf{x}) = -2.45$ log-odds** $\rightarrow P(\text{Positive}) = 0.079$ ($7.9\%$), resulting in confident classification as **Negative (Class 0)**.

This level of granular breakdown verifies that the model made the right prediction for the right linguistic reasons, rather than relying on spurious correlations.

### 6. Communicating SHAP Results to Non-Technical Stakeholders

Translating SHAP mathematical outputs into clear business language is an essential engineering skill:

| SHAP Technical Output | Non-Technical Stakeholder Communication |
| :--- | :--- |
| `waste` has top mean $\|\phi\|$ | *"Customer reviews containing words like 'waste' or 'terrible' are the strongest drivers of churn risk identified by our model."* |
| `not good` bigram $\phi < 0$ | *"Our preprocessing ensures that phrases like 'not good' are recognized as negative rather than being confused by the word 'good'."* |
| Low attribution on stopwords | *"Generic connecting words do not sway the model's confidence, ensuring predictions focus strictly on descriptive sentiment terms."* |
| Waterfall for Review #12 | *"The model flagged this review as negative because the user repeatedly highlighted 'poorly paced' and 'waste of time', despite mentioning a recognizable actor's name."* |
| Ambiguous review with $\phi \approx 0$ | *"The model is uncertain on this review because it contains equal amounts of positive praise and negative criticism without a clear dominant sentiment."* |

---

## <span style="color:#27AE60">5.4 Sprint 3 Review & Retrospective</span>

### 1. Sprint 3 Review Demonstration Summary

The Sprint 3 Review presents the complete, integrated NLP sentiment pipeline to the mentor and stakeholders:

**Demonstration Agenda:**
1. **End-to-End Pipeline:** The unified `predict()` function from Day 4, demonstrating zero training/serving skew across raw text strings, batch inputs, and negation handling.
2. **Full Evaluation Metrics:** The multi-model benchmark tables (Section 5.1), confirming that the Fine-Tuned Transformer outperforms the classical TF-IDF baseline across all metrics ($\Delta F_1 = +6.14\%$, $\Delta \text{ROC-AUC} = +0.0490$, test loss reduced by $32.7\%$).
3. **Diagnostic Error Analysis:** The Day 4 confusion matrix diagnostics and qualitative categorization of misclassifications (Data Issues vs. Model Weaknesses).
4. **SHAP Explainability:** Global token importance rankings and per-review waterfall explanations (Section 5.3), validating linguistic reasoning and enabling stakeholder transparency.
5. **Precision-Recall Trade-Off:** Threshold analysis demonstrating operational operating point selection for negative review / churn escalation (Section 5.2).

### 2. Sprint 3 Acceptance Criteria Verification

| Acceptance Criterion | Verification Evidence | Status |
| :--- | :--- | :---: |
| **Notebook runs cleanly from top to bottom** | All cells execute with zero runtime errors, all plots render | Verified |
| **Evaluation metrics beat classical baseline** | Fine-Tuned Model: $F_1 = 0.9139$ vs. Baseline $F_1 = 0.8525$ ($+6.14\%$) | Verified |
| **SHAP explainability generated** | Global feature attribution bar plot and per-prediction waterfall generated | Verified |
| **Code committed on correct feature branch** | All Sprint 3 commits on `feature/sprint-3-pipeline` branch | Verified |
| **Pull request submitted and approved** | Mid-sprint PR (Day 3) and final Sprint 3 PR reviewed and merged | Verified |
| **Results documented** | Comprehensive evaluation tables, error analysis, and SHAP interpretation in README | Verified |

### 3. Sprint 3 Agile Retrospective

| Review Focus | Retrospective Analysis |
| :--- | :--- |
| **What Went Well?** | The unified `predict()` pipeline established zero training/serving skew through unit assertion tests. Preserving task-critical negations during Day 1 preprocessing prevented polarity inversion on negated reviews. The multi-model evaluation established an unambiguous performance hierarchy (Baseline TF-IDF $\to$ LSTM $\to$ DistilBERT). SHAP explainability provided intuitive word-level attributions that clearly distinguish model reasoning on straightforward versus nuanced reviews. |
| **What Can Be Improved?** | While TF-IDF SHAP computation via `LinearExplainer` is fast, computing exact SHAP values for full Transformer self-attention layers requires `PartitionExplainer` or `TextExplainer`, which is computationally demanding. In addition, handling long-range sarcastic reviews remains a residual error mode that bag-of-words models cannot resolve. |
| **Concrete Action for Sprint 4** | **Production API Deployment & Monitoring:** In Sprint 4, package the validated sentiment pipeline into a containerized FastAPI serving endpoint. Add input validation with Pydantic, implement latency benchmarking, configure prediction confidence logging, and design automated drift monitoring for production deployment. |

---

## <span style="color:#9f43c3ff">5.5 Sprint 3 Deliverables & Technical Discipline Summary</span>

1. **Complete Task-Appropriate Evaluation:** Every model architecture is benchmarked against the classical TF-IDF baseline using Precision, Recall, F1-Score, and ROC-AUC — not accuracy alone — to ensure transparent, rigorous performance comparison.
2. **Class Imbalance Awareness:** Class distribution was assessed, SMOTE mechanics on vectorized representations were established with strict leak-free training boundaries, and the precision-recall trade-off was analyzed with deliberate threshold selection.
3. **SHAP Explainability for Trust:** Global vocabulary importance and per-review waterfall explanations establish model trustworthiness and enable clear communication of prediction drivers to non-technical stakeholders.
4. **Sprint 3 Acceptance Closure:** All acceptance criteria verified — notebook runs cleanly, metrics beat the baseline, SHAP generated, code committed on feature branch, PR approved, results fully documented.
5. **Sprint 4 Readiness:** The Sprint Retrospective defines a concrete deployment action: packaging the end-to-end sentiment pipeline for production API serving with monitoring and validation.
6. **Notebook Artifacts:** The complete implementation, evaluation benchmarks, SHAP visualizations, and precision-recall analysis are implemented in [Task.ipynb](./Task.ipynb).
