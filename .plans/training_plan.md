# ParagAI: Comprehensive Model Training Plan

> **Document:** Training Strategy, Data Pipeline, & Execution Roadmap  
> **Target:** Training the custom `ParagAI` Decoder-Only Transformer from scratch  
> **Location:** `.plans/training_plan.md`  
> **Status:** Ready for Execution  

---

## 1. Training Objective & Strategy

The goal is to train the **ParagAI** model from random weight initialization to a state where it generates grammatically sound, context-aware, and coherent text.

### The Self-Supervised Next-Token Prediction Objective
The model learns without manual data labeling by predicting the next token in any text sequence:
$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \dots, x_{t-1}; \theta)$$

Where:
* $x_1, \dots, x_{t-1}$ is the input context.
* $x_t$ is the ground-truth target token.
* $\theta$ represents the trainable weights of the neural network.

---

## 2. Dataset Strategy: Staged Progression

To ensure rapid debugging and cost-effective development, training is planned in 3 progressive stages:

```text
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ Stage 1: Sanity Check   │ ──► │ Stage 2: Proof of Concept│ ──► │ Stage 3: Domain Scale   │
│ Single Batch / Overfit  │     │ TinyShakespeare (~1 MB)│     │ TinyStories / Custom    │
│ Time: < 2 minutes       │     │ Time: 10 - 20 minutes   │     │ Time: 1 - 3 hours (GPU) │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

### Stage 1: The Zero-Loss Overfit Test (Sanity Check)
* **Dataset:** 5–10 sentences (e.g., a short paragraph).
* **Goal:** Verify that backpropagation, gradients, and attention are bug-free. The training loss must drop from $\approx 4.5$ to $< 0.1$.
* **Exit Criteria:** The model can perfectly reproduce the memorized sentences.

### Stage 2: Baseline Language Modeling (TinyShakespeare)
* **Dataset:** ~1.1 MB raw text (~300,000 words).
* **Goal:** Verify that the model learns words, punctuation, dialogue formatting, and basic syntax.
* **Exit Criteria:** Validation loss drops below $1.6$; generated text resembles coherent early modern English dialogue.

### Stage 3: Structured Coherent Storytelling (TinyStories or Custom Corpus)
* **Dataset:** 30–50 MB of simple, grammatically rich children's stories or domain-specific text (English or Bengali).
* **Goal:** Teach the model cross-sentence continuity, vocabulary richness, and logical storytelling.
* **Exit Criteria:** Validation loss $< 1.3$, low perplexity ($PPL < 4.0$).

---

## 3. Data Ingestion & Tokenization Pipeline

### 3.1 Data Preparation Steps
1. **Cleaning:** Remove non-printable characters, normalize whitespace, unify UTF-8 encoding.
2. **Splitting:** 
   - **Training Set (90%):** Used strictly for calculating gradients and updating weights.
   - **Validation Set (10%):** Kept strictly unseen during training to evaluate true generalization and detect overfitting.
3. **Tokenization:**
   - Convert raw strings into integer token arrays: $\text{text} \to [t_1, t_2, \dots, t_N]$.
   - Store as compact memory-mapped NumPy arrays (`uint16` / `int32`) for instant loading without RAM bottlenecks.

### 3.2 Dynamic Batch Slicing
* In each iteration, random chunk starting points $i$ are sampled from the token array:
  - **Inputs ($X$):** Slice tokens $[i : i + T]$
  - **Targets ($Y$):** Slice tokens $[i + 1 : i + T + 1]$
* Tensor dimensions:
  - $X \in \mathbb{R}^{B \times T}$
  - $Y \in \mathbb{R}^{B \times T}$
  *(where $B = \text{batch\_size}$ and $T = \text{max\_seq\_len}$)*

---

## 4. Hyperparameter & Training Configuration

| Hyperparameter | Value (Base / Small) | Value (Medium / Extended) | Reason |
|---|---|---|---|
| **Context Length ($T$)** | 256 tokens | 512 tokens | Length of memory/attention window |
| **Embedding Dim ($d_{\text{model}}$)** | 192 | 384 | Vector dimension per token |
| **Attention Heads ($n_{\text{head}}$)** | 6 | 6 | Multi-head subspace projections |
| **Transformer Layers ($n_{\text{layer}}$)** | 6 | 8 | Depth of the network |
| **Batch Size ($B$)** | 32 sequences | 64 sequences | Balances GPU/CPU RAM and gradient stability |
| **Peak Learning Rate** | $6.0 \times 10^{-4}$ | $4.0 \times 10^{-4}$ | Standard AdamW rate for small transformers |
| **Min Learning Rate** | $6.0 \times 10^{-5}$ | $4.0 \times 10^{-5}$ | $10\%$ of peak rate at end of cosine decay |
| **Warmup Steps** | 200 iterations | 500 iterations | Prevents early gradient explosion |
| **Total Iterations** | 5,000 steps | 15,000 steps | Complete training schedule |
| **Weight Decay** | 0.1 | 0.1 | $L_2$ regularization for non-norm weights |
| **Gradient Clipping** | 1.0 | 1.0 | Clips maximum $L_2$ gradient norm |
| **Dropout** | 0.1 | 0.1 | Prevents neuron co-adaptation |

---

## 5. Learning Rate Schedule: Cosine Annealing with Warmup

The learning rate $\eta_t$ at step $t$ follows two phases:
1. **Linear Warmup ($t < T_{\text{warmup}}$):**
   $$\eta_t = \eta_{\text{max}} \times \frac{t}{T_{\text{warmup}}}$$
2. **Cosine Decay ($T_{\text{warmup}} \le t \le T_{\text{max}}$):**
   $$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}}\pi\right)\right)$$

*Why this matters:* In the first few hundred steps, randomly initialized weights produce chaotic gradients. The linear warmup gently warms up the weights before full learning begins. The cosine decay helps the model settle into an optimal local minimum.

---

## 6. Training Execution Workflow

```
[ Step 1: Initialize Seeds & Device ]
        │ (Detect CPU or CUDA / MPS)
        ▼
[ Step 2: Build Tokenizer & Datasets ]
        │ (Fit vocabulary, split 90/10)
        ▼
[ Step 3: Instantiate ParagAI Model ]
        │ (Apply Gaussian weight initialization: std=0.02)
        ▼
[ Step 4: Configure Optimizer & Scheduler ]
        │ (AdamW with decoupled weight decay)
        ▼
┌────────────────────────────────────────────────────────┐
│               ITERATIVE TRAINING LOOP                  │
│                                                        │
│ 1. Sample batch (X, Y) from training set               │
│ 2. Forward pass: logits, loss = model(X, targets=Y)    │
│ 3. Backward pass: optimizer.zero_grad(), loss.backward│
│ 4. Clip gradients: torch.nn.utils.clip_grad_norm_      │
│ 5. Optimizer step: optimizer.step(), scheduler.step()  │
│                                                        │
│ Every 250 iterations:                                  │
│  - Compute Validation Loss over 50 batches             │
│  - Compute Perplexity: exp(val_loss)                   │
│  - Generate sample text prompt (visual sanity check)   │
│  - If val_loss is best so far -> Save checkpoint.pt    │
└────────────────────────────────────────────────────────┘
        │
        ▼
[ Step 5: Final Evaluation & Checkpoint Export ]
```

---

## 7. Hardware & Training Environments

### Option A: Local PC (CPU)
* Ideal for: Stage 1 (Sanity check) & Stage 2 (TinyShakespeare).
* Setup: Zero external configuration required. Runs directly with `python scripts/train.py --device cpu`.
* Expected duration: ~10 to 20 minutes.

### Option B: Local PC with NVIDIA GPU (CUDA)
* Ideal for: Fast iterations and Stage 3 datasets.
* Setup: Requires PyTorch with CUDA support.
* Expected duration: ~1 to 3 minutes for Stage 2.

### Option C: Free Cloud GPU (Google Colab / Kaggle)
* Ideal for: Users without a dedicated GPU who want 10x faster training.
* Setup: We will provide a self-contained single-command script or Jupyter Notebook (`notebooks/train_colab.ipynb`) that clones the repository and runs on a free T4 GPU.

---

## 8. Monitoring, Metrics & Checkpointing

### Key Metrics to Monitor
1. **Training Loss:** Should consistently trend downward.
2. **Validation Loss:** Tracks actual generalization.
   - If Validation Loss decreases along with Training Loss: **Model is learning well.**
   - If Training Loss drops but Validation Loss starts climbing: **Overfitting.** (Increase dropout or add more data).
3. **Perplexity ($PPL = e^{\text{Loss}}$):** Represents the effective number of words the model is uncertain about at any point. Lower is better.
   - Initial untrained state: $PPL \approx \text{Vocab Size}$ (e.g. 2000+)
   - Well-trained small model: $PPL < 10$

### Checkpoint Policy
Checkpoints will be saved automatically to `checkpoints/`:
* `checkpoint_latest.pt`: Saved every 500 steps (allows resuming interrupted training).
* `checkpoint_best.pt`: Saved only when validation loss reaches a new historical minimum.

---

## 9. Failure Modes & Troubleshooting

| Issue | Likely Cause | Fix |
|---|---|---|
| **Loss is `NaN` or `Inf`** | Learning rate too high, or exploding gradients. | Ensure gradient clipping (`grad_clip = 1.0`), reduce `learning_rate` to $3.0 \times 10^{-4}$. |
| **Loss does not decrease** | Learning rate too low, zero gradients, or improper weight init. | Check if optimizer step is running; verify normal initialization ($\sigma=0.02$). |
| **CUDA Out of Memory (OOM)** | Batch size or sequence length too large for VRAM. | Lower `batch_size` (e.g. from 32 to 16) or reduce `max_seq_len` (from 256 to 128). |
| **Repetitive / Gibberish Output** | Model untrained, or temperature is set to 0 with no penalty. | Train for more iterations; set `temperature=0.8` and `repetition_penalty=1.1`. |

---

## 10. Execution Commands Reference

```bash
# 1. Download and prepare the dataset
python scripts/prepare_data.py --dataset tinyshakespeare

# 2. Run sanity overfit test (verifies architecture in 60 seconds)
python scripts/train.py --config configs/sanity_test.yaml

# 3. Start standard training run
python scripts/train.py --config configs/train_config.yaml

# 4. Resume training from a saved checkpoint
python scripts/train.py --config configs/train_config.yaml --resume checkpoints/checkpoint_latest.pt

# 5. Interactive generation with the trained model
python scripts/chat.py --checkpoint checkpoints/checkpoint_best.pt --temperature 0.8
```

