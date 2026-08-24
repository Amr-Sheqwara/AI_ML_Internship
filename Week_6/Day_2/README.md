
<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 6 — Day 2: Activation Functions, Forward Propagation & Loss</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 2 Focus: Non-Linear Activations, Forward Propagation Mechanics & Loss Function Selection</b>

Today covers the core concepts of how neural networks compute predictions. We explore why non-linear activation functions are necessary to learn complex patterns, compare common activations (ReLU, Sigmoid, Tanh), justify the output activation and loss function for the Phase 3 Cardiac Patient Monitoring capstone project, and walk through a forward pass step-by-step.

</blockquote>

---

## <span style="color:#F78BA0">2.1 Overview & Key Objectives</span>

<blockquote style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">

A single neuron takes inputs, multiplies each by a weight, sums them, adds a bias, and passes the result through an activation function. A neural network is simply that same operation stacked and repeated across layers.

<b>Key Objectives:</b>
- Understand why activation functions introduce non-linearity and prevent deep networks from collapsing into simple linear models.
- Compare common activation functions: ReLU, Sigmoid, and Tanh across their output ranges and primary use cases.
- Choose and justify the correct output activation and loss function for the Phase 3 project task.
- Trace one complete forward pass through a 2-layer network on a sample input in NumPy.

</blockquote>

---

## <span style="color:#85C1E9">2.2 Why Activation Functions Matter</span>

Without an activation function, a neural network—no matter how many layers it has—collapses into a single linear model because a stack of linear operations is still linear. The activation function introduces non-linearity, allowing the network to learn complex, curved patterns from data.

| Activation Function | Output Range | Use It For | Key Behavior |
| :--- | :--- | :--- | :--- |
| **ReLU** | 0 to +infinity (negatives become 0) | **Hidden layers** | The standard default choice; fast, simple, and effective. |
| **Sigmoid** | 0 to 1 | **Output layer** | Used for binary classification to produce a probability. |
| **Tanh** | -1 to +1 | **Hidden layers** | Useful in hidden layers when zero-centered output is beneficial. |
| **Softmax** | 0 to 1 (all outputs sum to 1) | **Output layer** | Used for multi-class classification to produce class probabilities. |

**Practical Rule:** Use ReLU in hidden layers by default, and choose the output activation based on the task: Sigmoid for binary classification, Softmax for multi-class classification, and linear (no activation) for regression.

---

## <span style="color:#F8C471">2.3 Phase 3 Project Task: Activation & Loss Selection</span>

### 1. Task Description
- **Dataset:** Cardiac Patient Monitoring System (`heart.csv`)
- **Task:** Binary Classification predicting the target variable `HeartDisease` (0 = Normal, 1 = Heart Disease present).

### 2. Output Layer Activation Choice: Sigmoid
- **Justification:** Binary classification requires predicting a probability between 0 and 1. Sigmoid maps the final layer output into this range, representing the likelihood of heart disease for a given patient.

### 3. Loss Function Choice: Binary Cross-Entropy
- **Justification:** Binary Cross-Entropy is the standard loss function for binary classification tasks. It measures how far the predicted probability is from the true binary label (0 or 1) and penalizes confident incorrect predictions heavily to guide model learning.

---

## <span style="color:#309c42ff">2.4 Forward Pass Walkthrough & Summary</span>

Forward propagation is the process of data flowing through the network to produce a prediction:
1. **Input Layer:** Receives the raw feature values.
2. **Hidden Layer:** Computes weighted sums, adds biases, and applies the ReLU activation (clipping negative values to zero).
3. **Output Layer:** Computes the final weighted sum and applies Sigmoid to generate a prediction between 0 and 1.
4. **Loss Calculation:** Compares the prediction to the true label using Binary Cross-Entropy.

| Step | Stage | Value / Result | Description |
| :--- | :--- | :--- | :--- |
| **1** | Sample Input | `[0.50, -0.20]` | 2 patient feature inputs |
| **2** | Hidden Layer Pre-activation | `[0.42, -0.57]` | Weighted sum plus bias for each hidden neuron |
| **3** | Hidden Layer Output (ReLU) | `[0.42, 0.00]` | Negative neuron is deactivated by ReLU |
| **4** | Output Layer Pre-activation | `0.3440` | Weighted sum of hidden outputs plus bias |
| **5** | Final Prediction (Sigmoid) | `0.5852` | Predicted probability of heart disease |
| **6** | Binary Cross-Entropy Loss | `0.5359` | Error penalty evaluated against true label $y = 1$ |
