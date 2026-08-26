
<h1  style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">AI & ML Course with BinX — Week 6: Deep Learning Intro & Capstone Sprint 1</h1>

<blockquote  style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Phase 3 Milestone: Neural Networks, Backpropagation & Training with TensorFlow / Keras</b>

Welcome to Week 6 of the BinX Tech AI & ML Internship Program. This week marks the kickoff of Phase 3 and Sprint 1 of the applied capstone project. In alignment with agile sprint execution, interns work directly on the central Capstone Project repository rather than maintaining disconnected task files. Interns learn how neural networks are structured and trained—covering network architecture, non-linear activation functions, forward and backpropagation, gradient descent, optimizers, batch normalization, and dropout. We transition from classical models to building, training, tuning, and evaluating deep learning architectures in TensorFlow/Keras while executing the full Sprint 1 agile delivery cycle.

</blockquote>

---

## <span style="color:#F78BA0">Week Overview</span>

| Attribute | Details |
| :--- | :--- |
| **Curriculum Phase** | Phase 3: Deep Learning & Applied Project (Sprint 1 — 40 hrs) |
| **Week Focus** | Neural Network Architecture, Non-Linear Activations (ReLU, Sigmoid, Softmax, Tanh), Forward Propagation, Loss Functions, Backpropagation, Gradient Descent, Learning Rate, Optimizers (Adam, SGD), TensorFlow/Keras Sequential API, Batch Normalization, Dropout, Callbacks (EarlyStopping, ModelCheckpoint), Hyperparameter Tuning, Sprint 1 Execution |
| **Sprint Repository Strategy** | All code tasks are consolidated directly within the central Capstone Project notebook (`AI_ML_Internship/Projects/cardiac_monitoring_project/Notebooks/Cardiac_Patient_Monitoring_System.ipynb`) to maintain an integrated end-to-end clinical workflow. |
| **Documentation Deliverables** | Daily Sprint Documentation (`Day_1/README.md`, `Day_2/README.md`, `Day_3/README.md`, `Day_4/README.md`), Sprint 1 Baseline, Pull Requests, and Sprint Retrospective |
| **Tech Stack** | Python 3.10+, NumPy, Pandas, Matplotlib, TensorFlow / Keras, Scikit-Learn, Git & GitHub |

---

## <span style="color:#85C1E9">Sprint 1 Daily Task Structure & Roadmap</span>

| Day | Focus & Milestones | Key Sprint Deliverables | Documentation |
| :--- | :--- | :--- | :--- |
| **Day 1** | **Sprint 1 Kickoff & Baseline Model** | Sprint backlog prioritization; physiological data cleaning; leak-free pipeline; Logistic Regression baseline benchmark ($F_1 = 0.8986$, $\text{ROC-AUC} = 0.9329$). | [Day 1 README] |
| **Day 2** | **Activations, Forward Pass & Loss** | Non-linear activations comparison (ReLU, Sigmoid, Tanh, Softmax); output layer & Binary Cross-Entropy loss justification; analytical & NumPy 2-layer forward pass verification. | [Day 2 README] |
| **Day 3** | **Backpropagation & Learning Rates** | 4-stage optimization cycle; analytical gradients via chain rule; SGD vs. Adam dynamics; learning rate experiments ($\eta = 1.5, 0.0001, 0.1$) & loss curve diagnostics. | [Day 3 README] |
| **Day 4** | **Deep Networks in Keras (BatchNorm & Dropout)** | High-level Keras Sequential API; Compile/Fit/Evaluate workflow; Batch Normalization & Dropout regularization; training vs. validation loss curve diagnostics; baseline comparison benchmark ($F_1 = 0.9073$). | [Day 4 README] |

---

## <span style="color:#309c42ff">Week 6 Deliverables & Technical Discipline</span>

1.  **Baseline-First Principle:** A neural network score is meaningless without context. Every project sprint begins by training and evaluating a simple classical baseline model (from Weeks 3–4). The deep learning architecture only earns its place in the pipeline if it demonstrates clear empirical superiority over the baseline.

2.  **Loss Curve Diagnostics:** Never trust evaluation metrics without inspecting the training history. Plot training versus validation loss and accuracy across epochs to diagnose underfitting, overfitting, learning rate anomalies, and convergence stability before finalizing model choices.

3.  **Demystifying the Math:** Neural networks are not black boxes; they represent repeated matrix dot products, bias additions, and activation transformations. Backpropagation systematically computes exact analytical gradients using the chain rule to update weights in the steepest downhill direction on the loss surface.

4.  **Disciplined Hyperparameter Tuning:** Tune hyperparameters systematically in order of impact (learning rate $\rightarrow$ architecture width/depth $\rightarrow$ dropout $\rightarrow$ batch size), changing only one variable at a time. Rely on the Adam optimizer with default learning rates ($\approx  0.001$) and utilize `EarlyStopping` to prevent overtraining.

5.  **Engineering & Agile Rigor:** Ensure all notebooks run cleanly from top to bottom, maintain clean git branches, open structured pull requests for mentor review, document methodological choices clearly in Markdown, and record concrete action items in the Sprint Retrospective.