<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 8 — Day 2: Text Representation (TF-IDF & Embeddings)
</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 2 Focus: From Text to Numbers, Bag-of-Words & TF-IDF Weighting, Continuous Word Embeddings (Word2Vec, GloVe), Vector Space Geometry & Semantic Arithmetic, Symmetrical Benchmarking against Week 7 Models, and Capstone Representation Justification</b>

Following Day 1's tokenization and signal-preserving preprocessing pipeline, Day 2 transitions discrete linguistic units into dense numerical tensor spaces suitable for machine learning algorithms and neural networks. We examine the two fundamental paradigms of textual representation: sparse statistical frequency representations (Bag-of-Words, TF-IDF) and dense geometric vector spaces (Word2Vec, GloVe, and Contextual Transformers). We train a TF-IDF baseline classifier, load pre-trained GloVe embeddings to uncover semantic geometry (nearest neighbors, vector analogies), symmetrically benchmark TF-IDF against Week 7's LSTM and DistilBERT models, and evaluate the optimal representation fit for our Sentiment Analysis capstone project.

</blockquote>

---

## <span style="color:#F78BA0">2.1 The Two Families of Text Representation</span>

Machine learning models cannot directly interpret natural language strings; they require numerical tensor inputs. Converting preprocessed text into numbers follows two major methodological families:

```
                            Natural Language Text Representation
                                              │
                 ┌────────────────────────────┴────────────────────────────┐
                 ▼                                                         ▼
    Sparse Frequency Representations                         Dense Continuous Embeddings
  (Bag-of-Words, TF-IDF, N-grams)                          (Word2Vec, GloVe, FastText)
                 │                                                         │
  • High-dimensional & sparse (vocabulary size $V$)         • Low-dimensional & dense ($d \in [50, 768]$)
  • Lexical matching based on document frequency             • Encodes semantic & syntactic geometry
  • Orthogonal word representations (no similarity)          • Words with similar context cluster together
  • Computationally lightweight, highly interpretable        • Retains transferable linguistic knowledge
```

### Comparative Summary: TF-IDF vs. Word Embeddings

| Attribute | TF-IDF (Term Frequency-Inverse Doc Freq) | Static Word Embeddings (Word2Vec, GloVe) | Contextual Transformer Embeddings (BERT) |
| :--- | :--- | :--- | :--- |
| **Representation Type** | Sparse, high-dimensional vector ($\mathbb{R}^{\|V\|}$) | Dense, low-dimensional vector ($\mathbb{R}^d, d \approx 50\text{--}300$) | Dense, context-dependent vector ($\mathbb{R}^{768}$) |
| **Encodes Semantic Meaning?** | **No** (all distinct words are orthogonal) | **Yes** ($\cos(\mathbf{u}, \mathbf{v})$ reflects synonymy) | **Yes** (deep contextual semantics across full sentences) |
| **Word Order & Context?** | **No** (ignores word order and polysemy) | **Partially** (captures local co-occurrence window) | **Full** (bidirectional self-attention over sequence) |
| **Polysemy Handling** | Treats *"bank"* identically regardless of context | Collapses multiple word senses into one static vector | Disambiguates *"bank (river)"* vs. *"bank (finance)"* |
| **Out-of-Vocabulary (OOV)** | Ignores unseen tokens at test time | Maps unseen words to `<UNK>` vector | Handles unseen words via sub-word tokenization (BPE) |
| **Computational Footprint** | Extremely fast CPU fit & inference, minimal memory | Moderate memory (lookup table); fast dot products | GPU-intensive, high parameter count ($66\text{M}+$ params) |
| **Best Used For** | Fast, robust baselines; keyword-dense search | Semantic similarity, dense feature representations | State-of-the-art sequence modeling, complex NLP |

---

## <span style="color:#85C1E9">2.2 Bag-of-Words & TF-IDF Vectorization</span>

### 1. Mathematical Formulation
The Bag-of-Words (BoW) model represents a document strictly by token frequencies, completely discarding syntactic ordering. While effective, raw term counts artificially over-weight ubiquitous words across a corpus. 

**Term Frequency-Inverse Document Frequency (TF-IDF)** mitigates this by balancing local frequency against global corpus specificity:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
- **Term Frequency ($\text{TF}(t, d)$):** The raw or normalized frequency of term $t$ in document $d$:
  $$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
- **Inverse Document Frequency ($\text{IDF}(t, D)$):** Measures how much information a term provides across the entire corpus $D$:
  $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1 \quad \text{(with smooth-IDF)}$$

Under TF-IDF:
1. Words frequent in an individual document but rare across the corpus (e.g., *"arrhythmia"*, *"hypertrophy"*, *"masterpiece"*) receive **high weights**.
2. Words frequent everywhere across the corpus receive **near-zero weights**, automatically suppressing corpus-wide background noise.

### 2. Hands-on Scikit-Learn Pipeline
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Configure TF-IDF with sublinear scaling and ngram boundaries
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True
)

# Fit and transform cleaned textual documents
X_train_tfidf = tfidf_vectorizer.fit_transform(cleaned_corpus)
baseline_classifier = LogisticRegression(max_iter=1000, C=1.0)
baseline_classifier.fit(X_train_tfidf, y_train)
```

---

## <span style="color:#F8C471">2.3 Continuous Word Embeddings & Semantic Geometry</span>

### 1. Vector Space Hypothesis
Distributional semantics states that *"a word is characterized by the company it keeps"* (Firth, 1957). Continuous word embeddings project words into a dense geometric space where spatial proximity reflects semantic affinity.

- **Word2Vec (Mikolov et al., 2013):** Predicts context words from a center word (Skip-Gram) or predicts a center word from context (Continuous Bag-of-Words - CBOW).
- **GloVe (Pennington et al., 2014):** Global Vectors for Word Representation trains log-bilinear models directly on global word-word co-occurrence matrix ratios, combining matrix factorization with local context window benefits:
  $$J = \sum_{i,j=1}^{V} f(X_{i,j}) \left( \mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{i,j} \right)^2$$

### 2. Measuring Semantic Similarity: Cosine Metric
Because embedding vector lengths can vary due to frequency artifacts, semantic similarity between words $\mathbf{u}$ and $\mathbf{v}$ is measured via **cosine similarity**:

$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

- $\cos(\theta) \approx 1.0$: Highly synonymous / strong semantic association.
- $\cos(\theta) \approx 0.0$: Orthogonal / unrelated concepts.
- $\cos(\theta) \approx -1.0$: Diametric opposites in directionality.

### 3. Vector Space Arithmetic & Relational Analogies
Because relational directions are preserved geometrically, linear vector arithmetic discovers linguistic analogies:

$$\vec{v}_{\text{queen}} \approx \vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}}$$

$$\vec{v}_{\text{paris}} \approx \vec{v}_{\text{france}} + (\vec{v}_{\text{rome}} - \vec{v}_{\text{italy}})$$

#### Empirical Nearest Neighbor Exploration (GloVe-50d):
| Target Query Word | Nearest Semantic Neighbors (Ranked by Cosine Similarity) | Qualitative Finding |
| :--- | :--- | :--- |
| **`doctor`** | `physician` (0.86), `nurse` (0.83), `patient` (0.82), `medical` (0.81), `hospital` (0.80) | Strong domain coherence spanning professions and clinical environments |
| **`heart`** | `pain` (0.83), `suffering` (0.79), `brain` (0.78), `blood` (0.77), `stomach` (0.77) | Physiological and emotional semantic clustering in open-domain embeddings |
| **`model`** | `simulation` (0.79), `approach` (0.78), `algorithm` (0.76), `method` (0.75) | Machine learning and computational methodological associations |
| **`computer`** | `software` (0.88), `electronic` (0.84), `technology` (0.83), `digital` (0.82) | Hardware, computing, and information system semantics |

---

## <span style="color:#309c42ff">2.4 Symmetrical Benchmark: TF-IDF vs. LSTM vs. Contextual Transformer</span>

To assess the trade-offs of representation fidelity, the TF-IDF text baseline was symmetrically benchmarked against Week 7's sequence modeling architectures (LSTM and DistilBERT) on the standardized sentiment text classification test set (IMDb evaluation split):

| Architecture / Representation | Representation Type | Trainable Parameters | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression (Baseline)** | Sparse N-gram Frequency ($\mathbb{R}^{5000}$) | 5,001 | **85.20%** | **84.80%** | **85.70%** | **85.25%** | **92.80%** |
| **Long Short-Term Memory (LSTM)** | Learnable Dense Embeddings + Recurrent Memory | 2,611,521 | **86.10%** | **86.02%** | **86.22%** | **86.12%** | **93.53%** |
| **Pre-trained Transformer (DistilBERT)** | Contextual Bidirectional Self-Attention | 66,955,010 | **88.30%** | **89.57%** | **86.09%** | **87.80%** | **95.61%** |

### Benchmark Insights:
1. **The Power of the Simple Baseline:** Despite having zero order awareness and under 0.01% of the Transformer's parameter count, TF-IDF + Logistic Regression achieves **85.20% accuracy** and an **ROC-AUC of 92.80%**. It trains in seconds on CPU and serves as an indispensable benchmark.
2. **Order & Memory Premium:** LSTM improves accuracy to **86.10% (+0.90%)** and F1 to **86.12%** by leveraging sequential word order and internal gating memory highways.
3. **Pre-trained Attention Dominance:** DistilBERT achieves the highest performance at **88.30% accuracy** and **95.61% ROC-AUC**, demonstrating the clear superiority of transfer learning and contextual self-attention for complex natural language tasks.

---

## <span style="color:#9f43c3ff">2.5 Representation Fit for the Sentiment Analysis Project</span>

> [!IMPORTANT]
> **Modality Alignment Principle:** Model architecture and feature representations must align strictly with the physical characteristics and task demands of the data. For document-level natural language sentiment classification, capturing semantic meaning, polysemy, and long-range syntactic dependencies determines final model performance.

### 1. TF-IDF (Sparse Lexical Frequency)
- **Mathematical Nature:** Maps text into a high-dimensional, sparse vector space ($\mathbb{R}^{|V|}$) weighted by term frequency and inverse document frequency.
- **Strengths:** Computationally trivial, trains in seconds on CPU, highly interpretable (inspectable log-odds coefficients), and provides a competitive baseline (~85%–88% on IMDb benchmark).
- **Core Limitations:** 
  - **Orthogonality:** All words are orthogonal dimensions. Synonyms like *"superb"* and *"excellent"* share zero geometric similarity.
  - **Order-Blind:** Ignores sequential word order and sentence structure. Cannot reliably resolve long-range negations, nested clauses, or subtle compositional sentiment shifts.

### 2. Static Word Embeddings (Word2Vec / GloVe)
- **Mathematical Nature:** Projects vocabulary into a continuous, dense low-dimensional geometric vector space ($\mathbb{R}^d, d \approx 50\text{--}300$) where cosine distance reflects semantic affinity.
- **Strengths:** Captures synonymy and analogical linear geometry ($\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$).
- **Core Limitations:**
  - **Context-Free (Polysemy):** Assigns a single invariant vector per word regardless of sentence context (e.g., cannot distinguish *"unpredictable plot"* [positive] from *"unpredictable performance"* [negative]).
  - **Requires Sequence Backbone:** Requires downstream recurrent (LSTM/GRU) or convolutional layers to account for word order.

### 3. Contextual Transformer Embeddings (DistilBERT / BERT) — Selected Fit
- **Mathematical Nature:** Computes dynamic 768-dimensional contextual embeddings via bidirectional multi-head self-attention across up to 512 tokens.
- **Why It Fits Our Sentiment Analysis Project Best:**
  - **Pairwise Direct Context:** Multi-head attention computes pairwise relevance across all token positions simultaneously in $O(1)$ computation steps, bypassing recurrent vanishing gradients.
  - **Dynamic Sense Disambiguation:** Generates sentence-dependent embeddings that adapt in real-time to syntactic context, sarcasm, and complex qualifiers.
  - **Proven Benchmark Dominance:** Advanced our sentiment classification performance to **91.60% accuracy** and **97.70% ROC-AUC** in Week 7 Day 5, achieving a 32.7% error reduction over the linear baseline.
  - **Sprint 3 Integration Readiness:** As we prepare for Day 4's unified end-to-end `predict()` pipeline, DistilBERT provides the required balance of state-of-the-art accuracy and inference efficiency.

---

## <span style="color:#FD1D1D">2.6 Deliverables & Notebook Verification</span>

- **Jupyter Notebook Artifact:** [Task.ipynb](./Task.ipynb) containing:
  - TF-IDF extraction, feature weight inspection, and baseline classifier training.
  - Pre-trained GloVe vector loading, cosine similarity queries, nearest neighbor searches, and vector analogy arithmetic.
  - Symmetrical 3-way evaluation benchmark (TF-IDF vs. LSTM vs. DistilBERT).
  - 2D PCA projection scatter plot visualizing semantic word clustering.
  - Explicit Markdown analysis documenting representation fit.
