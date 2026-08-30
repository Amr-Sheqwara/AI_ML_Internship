
<h1  style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 7 — Day 1: Sprint 2 Planning & Convolutional Neural Networks
</h1>

<blockquote  style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 1 Focus: Sprint 2 Planning, Why Dense Networks Fail on Images, Convolution Foundations & Architecture Decision</b>

Today opens Phase 3, Sprint 2 of the applied capstone project. In this sprint, we advance the project's core model to beat the Week 6 baseline, carrying forward the process improvement from our Sprint 1 retrospective. Today we establish our Sprint 2 backlog, analyze the mathematical and computational limitations of fully connected dense networks on image data, examine how convolution operates via local dot products, filters/kernels, strides, and padding, explain parameter sharing and translation invariance, visualize feature maps via hand-defined edge detection filters, and formally record our project architecture selection based on data type.

</blockquote>  

---
  
## <span style="color:#F78BA0">1.1 Sprint 2 Planning & Backlog</span> 

<blockquote  style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">  

<b>Sprint 2 Goal:</b> Advance and tune the capstone project's core model using the architecture appropriate to its data modality, carry forward our Sprint 1 retrospective action, and decisively outperform the Week 6 baseline benchmark.

</blockquote>  

### Sprint 2 Core-Model Backlog Tasks:

1.  **Architecture Selection & Justification (Day 1):** Evaluate data modality (tabular vs. image vs. text/sequence) and select the matching core architecture.

2.  **Specialized Deep Learning Exploration (Days 1–4):**

- Convolutional mechanisms, pooling, and transfer learning for vision tasks.

- Recurrent architectures (RNN, LSTM, GRU) and embeddings for sequential data.

- Attention mechanisms and pre-trained Hugging Face Transformers for NLP.

3.  **Mid-Sprint Pull Request & Mentor Review (Day 3):** Open a pull request with the active notebook for structured mentor review.

4.  **Core Model Advancement & Tuning (Day 5):** Optimize network depth, width, regularization, and learning rate, logging all runs in MLflow.

5.  **Sprint 2 Close-Out & Retrospective (Day 5):** Produce benchmark comparison tables, present the Sprint Review demo, and record one concrete improvement for Sprint 3.  

### Retrospective Action Carried Forward from Sprint 1:

- Enforce strict baseline-first discipline and systematic experiment logging (tracking hyperparameters, loss curves, and evaluation metrics) before finalizing architectural enhancements.

---

## <span style="color:#85C1E9">1.2 Why Dense Networks Fail on Images</span>

Feeding high-dimensional spatial data (images) directly into fully connected (dense) networks presents two fundamental flaws:

1.  **Parameter Explosion & Computational Infeasibility:**

- A moderate $200  \times  200$ RGB color image contains $200  \times  200  \times  3 = 120,000$ values.

- Connecting this flattened input vector to a single dense hidden layer of only 128 neurons requires:

$$\text{Weights} = 120,000  \times  128 = 15,360,000  \text{ weights} + 128  \text{ biases} = 15,360,128  \text{ parameters}$$

- This causes extreme memory consumption, slow training, and severe overfitting.  

2.  **Destruction of Spatial Structure & Receptive Correlation:**

- In image data, nearby pixels are spatially correlated; a visual pattern (such as an edge, curve, or texture) is defined by local pixel arrangements.

- Dense layers flatten the 2D/3D image grid into a 1D vector, discarding all 2D spatial relationships and treating adjacent pixels identically to distant pixels.

- Dense networks lack location awareness: a pattern learned in the top-left corner cannot be recognized in the bottom-right corner without learning entirely new weights.

--- 

## <span style="color:#F8C471">1.3 Convolution: The Core Idea & Mathematical Mechanics</span>  

A convolution operation slides a small **filter (kernel)** — a tiny grid of learnable weights (e.g., $3  \times  3$) — across the input image, computing a localized dot product at each spatial position.

| Concept | Meaning & Operational Role |
| :--- | :--- |
| **Filter / Kernel** | A small grid of learnable weights that slides across the input to detect a specific local pattern (vertical edge, horizontal edge, curve, texture). |
| **Feature Map** | The 2D output grid showing the activation response and spatial location where the filter's pattern was detected. |
| **Stride** | The step size (number of pixels) the filter shifts horizontally and vertically across each step. |
| **Padding** | Adding border rows and columns (typically zeros) around the input image so the filter can process edge pixels and preserve output spatial dimensions. |

### Why Convolution Wins:

1.  **Parameter Sharing:** The same small filter is reused across every position of the entire image. A $3  \times  3$ RGB filter requires only $3  \times  3  \times  3 = 27$ weights (+ 1 bias), regardless of whether the image is $200  \times  200$ or $1024  \times  1024$.

2.  **Translation Invariance:** Because the identical filter slides across all coordinates, the feature map detects the pattern wherever it appears in the image.

3.  **Feature Hierarchy:** Early layers detect simple low-level patterns (edges, gradients), middle layers combine them into intermediate features (shapes, textures), and deep layers recognize high-level semantic objects (faces, objects).  

--- 

## <span style="color:#309c42ff">1.4 Hands-On Lab: Edge Detection & Parameter Analysis</span>  

### Step 2: Hand-Defined Convolution & Feature Map Visualization

 
Applying vertical and horizontal Sobel edge detection kernels to an image extracts directional intensity gradients.  

- A vertical filter produces large positive or negative dot products where pixel intensities change sharply from left to right (vertical boundary).

- A horizontal filter produces peak activations where pixel intensities change sharply from top to bottom (horizontal boundary).

### Step 3: Parameter Comparison (Dense vs. Convolutional Layer)

Assume an input image of size $200  \times  200  \times  3$ ($120,000$ inputs):

| Architecture Layer | Layer Specification | Parameter Calculation | Total Trainable Parameters |
| :--- | :--- | :--- | :--- |
| **Dense Layer** | 32 hidden units (fully connected) | $(120,000 \times 32) + 32$ | **3,840,032** |
| **Convolutional Layer** | 32 filters of size $3 \times 3 \times 3$ | $32 \times (3 \times 3 \times 3) + 32$ | **896** |  

**Conclusion:** The convolutional layer achieves a **4,285x parameter reduction** while preserving 2D spatial adjacency and providing translation invariance.
 
--- 

## <span style="color:#9f43c3ff">1.5 Step 4: Capstone Core Model Architecture Decision Record</span>


### Architecture Mapping Matrix:

| Project Data Type | Core Architecture |
| :--- | :--- |
| **Tabular (churn, house price, fraud, cardiac monitoring)** | **Dense network (Week 6) or gradient boosting; deep learning optional** |
| **Images (image classifier)** | CNN with transfer learning (Day 2) |
| **Text (sentiment analysis)** | LSTM or a pre-trained transformer (Days 3–4) |
| **Sequences / Time Series** | LSTM / GRU (Day 3) |

### Formal Decision Record:

-  **Selected Core Architecture:**  **Dense Neural Network (Multilayer Perceptron from Week 6)**.

-  **Justification:** Applying a CNN or RNN to tabular patient rows would impose artificial spatial or temporal priors where none exist. As defined in the Week 7 curriculum, matching the architecture to the data shape, rather than defaulting to the most complex option, is the mark of true applied machine learning.

-  **Sprint 2 Target:** Advance and tune the Week 6 Dense Neural Network (optimizing architecture depth/width, regularization, learning rates, and threshold tuning) to beat the Week 6 baseline benchmark.