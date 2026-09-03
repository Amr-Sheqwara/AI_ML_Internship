<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 7 — Day 5: Advancing the Core Model & Sprint 2 Review
</h1>

<blockquote style="border-left: 3px solid #85C1E9; padding-left: 12px; margin-left: 0;">

<b>Day 5 Focus: Symmetrical Baseline Benchmarking, In-Domain Transformer Advancement, Beating the Baseline & Sprint 2 Close-Out</b>

Today concludes Sprint 2 of the Phase 3 applied capstone cycle. In accordance with the program's acceptance criteria, we establish our empirical benchmark against a classical text classification baseline, advance our project architecture to an in-domain pre-trained Transformer (DistilBERT-IMDb), confirm that the advanced core model decisively beats the baseline across all primary evaluation metrics, and complete the formal Sprint 2 Review and Agile Retrospective.

</blockquote>

---

## <span style="color:#85C1E9">5.1 Choosing the Right Architecture for the Project</span>

Per the **Architecture-to-Data Alignment Principle**, our core model selection directly reflects the mathematical properties of document-level sentiment classification across lengthy movie reviews:

* **Classical Baseline (TF-IDF + Logistic Regression):**
  Constructs a global n-gram term frequency-inverse document frequency matrix across the entire document. It provides a strong, convex linear benchmark by evaluating document-wide lexical polarity without sequence length truncation.
* **Selected Core Architecture (In-Domain Pre-Trained Transformer - DistilBERT):**
  Leverages deep multi-head self-attention with a 512-token context window, pre-trained on open-domain corpora and fine-tuned specifically on in-domain IMDb review structures (`lvwerra/distilbert-imdb`). It eliminates the $O(n)$ recurrent memory bottleneck while resolving domain-transfer probability entropy.

---

## <span style="color:#85C1E9">5.2 Advancing the Core Model Over the Baseline</span>

Sprint 2's primary deliverable is a meaningfully improved core model over Sprint 1, with its empirical scores clearly beating the baseline under leak-free, symmetrical test conditions.

Both models were evaluated on the held-out IMDb test split under identical evaluation parameters ($N = 1000$ test reviews, `random_state=42`):

### Comparative Benchmark Results:

| Model Milestone | Architecture | Trainable Parameters | Test Loss | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | Status vs. Baseline |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Classical Baseline** | Logistic Regression | 20,001 | 0.3172 | 88.80 | 88.55 | 88.55 | 88.55 | 95.57 | Baseline Anchor |
| **2. Selected Core Model** | DistilBERT-IMDb | 66,955,010 | **0.2134** | **91.60** | **91.58** | **91.21** | **91.39** | **97.70** | **+2.80% (Beats Baseline)** |

---

## <span style="color:#85C1E9">5.3 Technical & Empirical Takeaways</span>

1. **Decisively Beating the Baseline:**
   - The advanced in-domain Transformer achieved **91.60% Accuracy** (+2.80% over the baseline) and **97.70% ROC-AUC** (+2.13% over the baseline).
   - Test cross-entropy loss dropped from 0.3172 to **0.2134**, achieving a **32.7% error reduction** compared to the classical linear baseline.
2. **Context Window & Bidirectional Self-Attention:**
   - The baseline relies strictly on independent bag-of-words presence. It cannot resolve compositional shifts, sarcasm, or nested negations (e.g., *"Not that the film was boring, but it certainly lacked depth"*).
   - The Transformer's scaled dot-product attention computes pairwise relevance across all token positions simultaneously in $O(1)$ computation steps:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
     This allows contextual representations to capture syntactic clause boundaries across the 512-token sequence.
3. **In-Domain Fine-Tuning Calibration:**
   - Zero-shot models trained on short sentence snippets (such as SST-2) suffer from distributional shift on multi-paragraph documents.
   - Adapting the attention weights directly to IMDb document structures aligns probability outputs, eliminating prediction entropy near classification thresholds.

---

## <span style="color:#309c42ff">5.4 Sprint 2 Review & Agile Retrospective</span>

### 1. Sprint 2 Review Demonstration Summary
* **Demonstration Objective:** Present the advanced core model and empirical evidence to the mentor, confirming that the model beats the baseline.
* **Sprint Acceptance Verification:**
  - The notebook executes cleanly from top to bottom with zero errors.
  - Symmetrical held-out evaluation was maintained across all experiments.
  - The in-domain Transformer surpasses the classical baseline across Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Cross-Entropy Loss.
  - All Sprint 2 backlog deliverables are complete. Zero tasks roll over into Sprint 3.

### 2. Sprint 2 Agile Retrospective

| Review Focus | Retrospective Analysis |
| :--- | :--- |
| **What Went Well?** | - Symmetrical benchmarking protocol provided transparent visibility into model performance.<br>- Identifying the domain-shift root cause enabled an immediate empirical gain from 88.30% to 91.60%.<br>- Experiment logging verified that deep attention provides genuine margin gains over linear baselines when properly calibrated. |
| **What Can Be Improved?** | - Full document evaluation is still constrained by BERT's 512-token maximum sequence length due to $O(n^2)$ memory scaling.<br>- Inference latency on CPU requires batching and quantization for real-time operational serving. |
| **Concrete Action for Sprint 3** | **Implement Long-Context Processing and Model Quantization:** In Sprint 3, evaluate document-chunk pooling or linear attention mechanisms to handle 1,000+ word reviews without truncation, and apply INT8 dynamic quantization to accelerate CPU inference latency. |
