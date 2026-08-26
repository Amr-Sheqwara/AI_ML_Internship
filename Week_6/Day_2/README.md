
<h1  style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 6 — Day 2: Activation Functions, Forward Propagation & Loss
</h1>  

<blockquote  style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">  

<b>Day 2 Focus: Non-Linear Activations, Forward Propagation Mechanics & Loss Function Selection</b>

Today covers the core mechanisms of how neural networks compute predictions. We explore why non-linear activation functions are necessary to learn complex curved patterns, compare common activations (ReLU, Sigmoid, Tanh, Softmax), justify the output activation and loss function for the Phase 3 Cardiac Patient Monitoring capstone project, and walk through a full forward pass step-by-step using both analytical math and NumPy.  

</blockquote>  

---  

## <span style="color:#F78BA0">2.1 Overview & Key Objectives</span>  

<blockquote  style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">  

A single neuron takes inputs, multiplies each by a weight, sums them, adds a bias, and passes the result through an activation function. A neural network is simply that same operation stacked and repeated across layers. Forward propagation is the network running forward to compute a prediction, representing the first half of every training step.  

<b>Key Objectives:</b> 

- Understand why activation functions introduce non-linearity and prevent deep networks from collapsing into simple linear models.

- Compare common activation functions: ReLU, Sigmoid, Tanh, and Softmax across their output ranges, gradients, and primary use cases.

- Choose and justify the correct output activation (Sigmoid) and loss function (Binary Cross-Entropy) for the Phase 3 project task.

- Trace one complete forward pass through a 2-layer network on a sample input in NumPy and analytically.

- Integrate the forward pass derivations into Section 3.1 of the Capstone Project notebook. 

</blockquote>

---  

## <span style="color:#85C1E9">2.2 Why Activation Functions Matter</span>  

Without an activation function, a neural network—no matter how many layers it has—collapses into a single linear model because a stack of linear operations is still linear. The activation function introduces non-linearity, which is what allows the network to learn complex, curved decision boundaries from data.  

| Activation Function | Output Range | Use It For | Key Mathematical Behavior |
| :--- | :--- | :--- | :--- |
| **ReLU** ($\max(0, z)$) | $0$ to $+\infty$ (negatives become $0$) | **Hidden layers** | Standard default choice; computationally efficient, avoids vanishing gradients for positive inputs. |
| **Sigmoid** ($\frac{1}{1 + e^{-z}}$) | $0$ to $1$ | **Output layer** | Used for binary classification to map outputs to a valid probability $\hat{y} \in (0, 1)$. |
| **Tanh** ($\frac{e^z - e^{-z}}{e^z + e^{-z}}$) | $-1$ to $+1$ | **Hidden layers** | Zero-centered output; helpful when negative activations carry meaningful contrast. |
| **Softmax** ($\frac{e^{z_i}}{\sum_j e^{z_j}}$) | $0$ to $1$ (sums to $1$) | **Output layer** | Used for multi-class classification to produce a normalized probability distribution. |  

**Practical Rule:** Use ReLU in hidden layers by default, and choose the output activation based on the task: Sigmoid for binary classification, Softmax for multi-class classification, and linear (no activation) for regression.  

--- 

## <span style="color:#F8C471">2.3 Phase 3 Project Task: Activation & Loss Selection</span>  

### 1. Task Description

-  **Dataset:** Cardiac Patient Monitoring System (`heart.csv`)

-  **Task:** Binary Classification predicting the target variable `HeartDisease` ($0$ = Normal, $1$ = Heart Disease present).  

### 2. Output Layer Activation Choice: Sigmoid

-  **Justification:** Binary classification requires predicting a calibrated probability between $0$ and $1$. Sigmoid maps the final linear combination into this range, representing the patient's likelihood of cardiac disease ($\hat{y} = P(\text{HeartDisease} = 1  \mid  \mathbf{x})$).  

### 3. Loss Function Choice: Binary Cross-Entropy

-  **Mathematical Formulation:**

$$\mathcal{L}_{BCE}(y, \hat{y}) = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

-  **Justification:** Binary Cross-Entropy is the standard maximum likelihood loss function for binary targets. It quantifies how far the predicted probability is from the true label ($0$ or $1$) and imposes heavy logarithmic penalties on confident incorrect predictions.  

---  

## <span style="color:#309c42ff">2.4 Forward Pass Walkthrough & Verification</span>

Forward propagation is the process of data flowing forward through the network layer-by-layer to produce a prediction:

1.  **Input Layer:** Receives the feature vector $\mathbf{x} = [0.50, -0.20]$.

2.  **Hidden Layer:** Computes weighted sum $\mathbf{z}_1 = \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1$, adds biases, and applies the ReLU activation $\mathbf{a}_1 = \max(0, \mathbf{z}_1)$.

3.  **Output Layer:** Computes linear combination $z_2 = \mathbf{a}_1  \mathbf{W}_2 + b_2$ and applies Sigmoid $\hat{y} = \sigma(z_2)$.

4.  **Loss Evaluation:** Evaluates $\hat{y}$ against true clinical label $y = 1.0$ using Binary Cross-Entropy.

| Step | Stage | Value / Result | Description |
| :--- | :--- | :--- | :--- |
| **1** | Sample Input ($\mathbf{x}$) | `[0.50, -0.20]` | 2 patient feature inputs |
| **2** | Hidden Pre-activation ($\mathbf{z}_1$) | `[0.42, -0.57]` | Weighted sum plus bias for hidden neurons |
| **3** | Hidden Activation ($\mathbf{a}_1$) | `[0.42, 0.00]` | Negative neuron is deactivated by ReLU ($\max(0, z)$) |
| **4** | Output Pre-activation ($z_2$) | `0.3440` | Weighted sum of active hidden representations plus bias |
| **5** | Final Prediction ($\hat{y}$) | `0.5852` | Predicted probability of heart disease ($\sigma(0.3440)$) |
| **6** | Binary Cross-Entropy Loss ($\mathcal{L}$) | `0.5359` | Error penalty evaluated against true label $y = 1.0$ | 

---

## <span style="color:#9f43c3ff">2.5 Sprint Integration & Deliverables</span>

-  **Capstone Notebook:** (`AI_ML_Internship/Projects/cardiac_monitoring_project/Notebooks/Cardiac_Patient_Monitoring_System.ipynb`) (Section 3.1: Activations, Derivative Visualizations, and Analytical Forward Pass).

-  **Code Deliverable:** Analytical and NumPy implementation of activation curves, derivatives, and forward propagation mechanics.