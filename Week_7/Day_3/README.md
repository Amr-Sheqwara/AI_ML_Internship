<h1 style="color:#C39BD3; border-bottom: 3px solid #C39BD3; padding-bottom: 8px;">
AI & ML Course with BinX — Week 7 — Day 3: RNNs & LSTMs for Sequential Data
</h1>

<blockquote style="border-left: 3px solid #C39BD3; padding-left: 12px; margin-left: 0;">

<b>Day 3 Focus: Sequential Data Properties, Recurrent Hidden States, The Vanishing Gradient Problem, Gated Memory Cells (LSTM) & Symmetrical Benchmarking</b>

Today covers sequence modeling and specialized recurrent architectures for sequential natural language processing. We examine why sequential data depends on word order; convert discrete word tokens into continuous geometric vectors via word embeddings; construct plain Recurrent Neural Networks (SimpleRNN) to analyze hidden state memory passing; explain why vanishing gradients cause plain RNNs to forget early context over long sequences; implement Long Short-Term Memory (LSTM) networks with gated memory cells; and benchmark LSTM against SimpleRNN on the IMDb dataset.

</blockquote>

---

## <span style="color:#F78BA0">3.1 Sequential Data Properties & Order Dependency</span>

Sequential data (such as text, time series, and audio) contains **order dependencies** where the arrangement of words defines semantic meaning.

| Concept | Description & Operational Impact |
| :--- | :--- |
| **Order Dependency** | Changing word order completely alters meaning (e.g., *"not bad, really good"* versus *"not good, really bad"*). |
| **Variable Sequence Length** | Reviews naturally differ in word count, requiring uniform padding and truncation. |
| **Long-Range Context** | Key sentiment context appearing early in a review must be remembered across hundreds of subsequent words. |

### Text Preprocessing & Embedding Pipeline:

1. **Text Cleaning & Tokenization:**
   - HTML tags (`<br />`) and excess whitespaces are stripped.
   - A `Tokenizer` builds a vocabulary dictionary of the top 20,000 most frequent words (`VOCAB_SIZE = 20,000`), reserving index 1 for unknown tokens (`<OOV>`).

2. **Sequence Padding (`pad_sequences`):**
   - Reviews are standardized to a fixed length of 300 words (`MAX_LEN = 300`).
   - Pre-padding (`padding="pre"`) places zeros at the beginning so the final emitted hidden state directly captures the closing sentiment words.

3. **Continuous Word Embeddings (`Embedding` Layer):**
   - Sparse word IDs are projected into dense 128-dimensional continuous vector space (`EMBEDDING_DIM = 128`).
   - Words with similar meanings learn nearby geometric positions during training.

---

## <span style="color:#85C1E9">3.2 Plain Recurrent Neural Networks (SimpleRNN)</span>

A plain Recurrent Neural Network processes words step-by-step ($t = 1, \dots, T$), updating a recurrent **hidden state** ($h_t$) that serves as working memory:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

- $x_t$: Current word embedding vector at time step $t$.
- $h_{t-1}$: Hidden state passed from the previous step.
- $h_t$: New hidden state carrying accumulated memory forward.
- $\tanh$: Activation function bounding the memory state between $-1$ and $1$.

### Parameter Sharing:
The same weight matrices ($W_h, W_x$) and bias ($b$) are reused at every step, allowing the network to process text of any length without increasing parameter count.

---

## <span style="color:#F8C471">3.3 The Vanishing Gradient Problem</span>

During Backpropagation Through Time (BPTT), error gradients must flow backward from the final step ($T = 300$) to early time steps ($t = 1$).

### Why Plain RNNs Fail on Long Sequences:
1. **Repeated Multiplication:** At every backward step, the gradient is multiplied by the recurrent weight matrix ($W_h$) and the derivative of $\tanh$ (which is always $\le 1$).
2. **Exponential Decay:** Multiplying numbers smaller than 1 across 300 consecutive steps causes gradients to shrink exponentially toward zero.
3. **Memory Loss:** Because gradients vanish, early words receive virtually no weight updates. The SimpleRNN effectively forgets the beginning of long reviews and makes predictions based only on the last few words.

---

## <span style="color:#309c42ff">3.4 Long Short-Term Memory (LSTM) Networks</span>

LSTMs solve vanishing gradients by maintaining a separate, additive memory channel called the **cell state** ($C_t$), regulated by three internal gates.

### Core LSTM Gating Mechanisms:

| Gate / Component | Formula | Operational Purpose |
| :--- | :--- | :--- |
| **Forget Gate ($f_t$)** | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ | Decides what percentage of old memory ($C_{t-1}$) to discard ($0 = \text{erase}, 1 = \text{keep}$). |
| **Input Gate ($i_t$)** | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ | Decides which new candidate values ($\tilde{C}_t$) to store in memory. |
| **Candidate State ($\tilde{C}_t$)** | $\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$ | Creates new candidate information from the current word. |
| **Cell State Update ($C_t$)** | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ | **Additive Memory Highway:** Combines kept past memory with new candidate memory. |
| **Output Gate ($o_t$)** | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ | Decides which parts of the cell state to emit as the hidden state. |
| **Hidden State ($h_t$)** | $h_t = o_t \odot \tanh(C_t)$ | Emits filtered memory to the next layer and time step. |

### Why LSTMs Remember Long Context:
The cell state update is **additive** ($+$), not multiplicative. When the forget gate is open ($f_t \approx 1$), gradients flow backward across hundreds of time steps without decaying, allowing the LSTM to retain early context throughout 300-word sequences.

---

## <span style="color:#F8C471">3.5 Symmetrical Benchmark Lab: LSTM vs. Plain SimpleRNN</span>

Both architectures were evaluated under identical conditions on the 25,000-sample IMDb test set (`VOCAB_SIZE=20000`, `MAX_LEN=300`, `EMBEDDING_DIM=128`, `BATCH_SIZE=64`).

### Comparative Performance Results:

| Architecture | Order-Aware | Gated Memory | Parameters | Test Loss | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plain SimpleRNN** | Yes | No | 2,574,465 | 0.3746 | 84.44 | 88.59 | 79.06 | 83.56 | 91.79 |
| **LSTM** | Yes | Yes | 2,611,521 | **0.3262** | **86.10** | 86.02 | **86.22** | **86.12** | **93.53** |

### Key Takeaways:

1. **Lower Prediction Error:** LSTM reduced test loss from `0.3746` to `0.3262`, indicating significantly better confidence and lower entropy.
2. **Elimination of Recall Deficit:** SimpleRNN dropped to `79.06%` recall due to forgetting early positive sentiment context. LSTM balanced precision (`86.02%`) and recall (`86.22%`), achieving an F1-Score of `86.12%`.
3. **Superior Discrimination:** LSTM achieved `93.53%` ROC-AUC versus `91.79%` for SimpleRNN across classification thresholds.
4. **Faster Training:** LSTM ran in **~7 seconds per epoch** (leveraging optimized C++ operations), while SimpleRNN required **~60 to 71 seconds per epoch**.
