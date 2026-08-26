
<h1  style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 6 — Day 3: Backpropagation, Gradient Descent & Learning Rates
</h1>

<blockquote  style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 3 Focus: 4-Stage Optimization Loop, Backpropagation via Chain Rule & Learning Rate Dynamics</b>
  
Today covers how neural networks learn from errors. We examine the 4-stage training cycle (Forward Pass, Loss Computation, Backpropagation, and Parameter Updates), the mathematical role of the multivariate chain rule in calculating exact analytical gradients, and the empirical diagnosis of learning rate regimes (Too High, Too Low, and Optimal) using loss and accuracy curves on the Cardiac Patient Monitoring dataset.  

</blockquote>
  
---
  
## <span style="color:#F78BA0">3.1 Overview & Key Objectives</span>
  
<blockquote  style="border-left: 3px solid #F78BA0; padding-left: 12px; margin-left: 0;">  

Neural networks optimize their weights iteratively. In each training cycle, predictions are generated, error penalties are computed, analytical gradients are derived backward through the computational graph, and parameters are adjusted along the steepest downhill direction on the loss surface.  

<b>Key Objectives:</b>  

- Master the 4-step training cycle: Forward Pass, Loss Evaluation, Backpropagation via the Chain Rule, and Weight Updates.

- Build and train a Multi-Layer Perceptron (MLP) in TensorFlow/Keras using the Sequential API for cardiac diagnosis.

- Systematically evaluate Stochastic Gradient Descent (SGD) across three learning rate regimes ($\eta = 1.5$, $\eta = 0.0001$, and $\eta = 0.1$).

- Diagnose convergence stability, underfitting, and learning rate anomalies via training and validation loss/accuracy curves.

- Complete the mid-sprint code and notebook review deliverables within the central Capstone Project repository.  

</blockquote>

---  

## <span style="color:#85C1E9">3.2 The 4-Stage Neural Network Training Cycle</span>  

| Stage | Operation | Mathematical Representation | Description |
| :--- | :--- | :--- | :--- |
| **[1] Forward Pass** | Prediction Generation | $z = \mathbf{w} \cdot \mathbf{x} + b$<br>$\hat{y} = \sigma(z)$ | Data flows forward layer-by-layer; computes linear combinations, adds biases, and applies non-linear activations (ReLU in hidden layers, Sigmoid in output). |
| **[2] Compute Loss** | Error Quantification | $\mathcal{L}_{BCE}(y, \hat{y}) = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | Evaluates the model prediction against the true label, producing a scalar error penalty. |
| **[3] Backpropagation** | Gradient Computation | $\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}}$ | Traverses backward through the computational graph using the **multivariate chain rule** to calculate exact analytical gradients with respect to all weights and biases. |
| **[4] Parameter Update** | Gradient Descent | $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{w}}$ | Adjusts weights and biases in the opposite direction of the gradient scaled by the **learning rate ($\eta$)** to minimize loss. |  

---  

## <span style="color:#F8C471">3.3 Why the Chain Rule is Involved</span> 

A neural network is a composition of nested functions across multiple layers. Because the loss $\mathcal{L}$ depends on the output activation $\hat{y}$, which depends on the linear combination $z$, which in turn depends on the weights $\mathbf{w}$, computing the sensitivity of the loss to any individual weight requires the multivariate chain rule:  

$$\frac{\partial  \mathcal{L}}{\partial  \mathbf{w}} = \frac{\partial  \mathcal{L}}{\partial  \hat{y}} \cdot  \frac{\partial  \hat{y}}{\partial z} \cdot  \frac{\partial z}{\partial  \mathbf{w}}$$  

For intermediate hidden layers, the error gradients are propagated recursively from the output layer back to the input layer:  

$$\frac{\partial  \mathcal{L}}{\partial  \mathbf{w}^{(l)}} = \boldsymbol{\delta}^{(l)} \cdot (\mathbf{a}^{(l-1)})^T, \quad  \text{where } \boldsymbol{\delta}^{(l)} = ((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}) \odot  \sigma'(\mathbf{z}^{(l)})$$ 

This systematically decomposes the global optimization problem into local partial derivatives calculated step-by-step from output to input.

---  

## <span style="color:#309c42ff">3.4 Learning Rate Dynamics & Diagnostic Summary</span>  

Training experiments were conducted on the Cardiac Patient Monitoring dataset using a 2-hidden-layer MLP architecture (`Dense(16, relu)` $\rightarrow$  `Dense(8, relu)` $\rightarrow$  `Dense(1, sigmoid)`) trained with SGD over 80 epochs:  

| Learning Rate Regime | Setting ($\eta$) | Observed Training Behavior | Diagnostic Signature |
| :--- | :--- | :--- | :--- |
| **Too High** | $\eta = 1.5$ | Overshoots local minima; causes volatile loss oscillations and instability. | Large spikes and erratic fluctuations in loss curves; fails to converge smoothly. |
| **Too Low** | $\eta = 0.0001$ | Step sizes are too small; model fails to make meaningful progress within 80 epochs. | Flat loss curve; severe underfitting and stagnation near initial loss values ($\approx 0.69$). |
| **Good / Optimal** | $\eta = 0.1$ | Smooth, steady gradient descent steps toward minimum loss. | Monotonically decreasing loss curve converging smoothly with steady accuracy gains ($> 85\%$). |

---
 
## <span style="color:#9f43c3ff">3.5 Sprint Integration & Deliverables</span> 

-  **Capstone Notebook:** (`AI_ML_Internship/Projects/cardiac_monitoring_project/Notebooks/Cardiac_Patient_Monitoring_System.ipynb`) (Section 3.2: The 4-Stage Optimization Cycle, Learning Rate Experiments, and Training Diagnostics).

-  **Code Deliverable:** Multi-Layer Perceptron model definition, SGD learning rate comparison loop, and Matplotlib loss/accuracy diagnostic curves.