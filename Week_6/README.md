
<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">AI & ML Course with BinX — Week 6: Deep Learning Intro & Capstone Sprint 1</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Phase 3 Milestone: Neural Networks, Backpropagation & Training with TensorFlow / Keras</b>

Welcome to Week 6 of the BinX Tech AI & ML Internship Program. This week marks the kickoff of Phase 3 and Sprint 1 of the applied capstone project. Interns learn how neural networks are structured and trained—covering network architecture, non-linear activation functions, forward and backpropagation, gradient descent, optimizers, batch normalization, and dropout. We transition from classical models to building, training, tuning, and evaluating deep learning architectures in TensorFlow/Keras while executing the full Sprint 1 agile delivery cycle.

</blockquote>

---

## <span style="color:#F78BA0">Week Overview</span>

| Attribute | Details |
| :--- | :--- |
| **Curriculum Phase** | Phase 3: Deep Learning & Applied Project (Sprint 1 — 40 hrs) |
| **Week Focus** | Neural Network Architecture, Non-Linear Activations (ReLU, Sigmoid, Softmax, Tanh), Forward Propagation, Loss Functions, Backpropagation, Gradient Descent, Learning Rate, Optimizers (Adam, SGD), TensorFlow/Keras Sequential API, Batch Normalization, Dropout, Callbacks (EarlyStopping, ModelCheckpoint), Hyperparameter Tuning, Sprint 1 Execution |
| **Deliverables** | Daily Jupyter Notebooks (`Task.ipynb`), Daily Documentation (`README.md`), Sprint 1 Baseline, Pull Requests, and Sprint Retrospective |
| **Tech Stack** | Python 3.10+, NumPy, Pandas, Matplotlib, TensorFlow / Keras, Scikit-Learn, Git & GitHub |

---

## <span style="color:#309c42ff">Week 6 Deliverables & Technical Discipline</span>

1. **Baseline-First Principle:** A neural network score is meaningless without context. Every project sprint begins by training and evaluating a simple classical baseline model (from Weeks 3–4). The deep learning architecture only earns its place in the pipeline if it demonstrates clear empirical superiority over the baseline.

2. **Loss Curve Diagnostics:** Never trust evaluation metrics without inspecting the training history. Plot training versus validation loss and accuracy across epochs to diagnose underfitting, overfitting, learning rate anomalies, and convergence stability before finalizing model choices.

3. **Demystifying the Math:** Neural networks are not black boxes; they represent repeated matrix dot products, bias additions, and activation transformations. Backpropagation systematically computes exact analytical gradients using the chain rule to update weights in the steepest downhill direction on the loss surface.

4. **Disciplined Hyperparameter Tuning:** Tune hyperparameters systematically in order of impact (learning rate $\rightarrow$ architecture width/depth $\rightarrow$ dropout $\rightarrow$ batch size), changing only one variable at a time. Rely on the Adam optimizer with default learning rates ($\approx 0.001$) and utilize `EarlyStopping` to prevent overtraining.

5. **Engineering & Agile Rigor:** Ensure all notebooks run cleanly from top to bottom, maintain clean git branches, open structured pull requests for mentor review, document methodological choices clearly in Markdown, and record concrete action items in the Sprint Retrospective.
