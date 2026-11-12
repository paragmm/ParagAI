# ParagAI: Project Plan — Custom AI Model from Scratch

> **Status:** Draft / Active  
> **Target:** Simple, modular, professional AI Model implemented in Python from first principles  
> **Workspace:** `ParagAI`  
> **Version:** 1.0.0  

---

## 1. Executive Summary & Vision

The objective of **ParagAI** is to build a modern, elegant, and production-grade AI model from scratch in Python. Rather than treating AI as a "black box" by simply wrapping third-party APIs, this project builds the end-to-end architecture from the ground up:

1. **First-Principles Understanding:** From mathematical foundations (linear algebra, backpropagation, self-attention) to tensor implementation.
2. **Dual-Tier Implementation:**
   - **Tier 1 (Core Foundations):** Pure NumPy micro-autograd engine and multi-layer perceptron (understanding computational graphs, gradients, and optimization).
   - **Tier 2 (Modern Architecture):** A clean, modular Decoder-Only Transformer (Mini-LLM / GPT-style) built with modern standards (PyTorch as tensor/GPU backend, custom-implemented attention, tokenization, training, and inference loops).
3. **Professional Software Engineering:** Production structure with typing, configuration management, automated testing (`pytest`), reproducible seeds, checkpointing, evaluation benchmarks, and a serving API (CLI & FastAPI).

---

## 2. Technical Architecture & Design Decisions

### 2.1 Model Architecture (Decoder-Only Transformer)
* **Architecture:** Autoregressive Decoder-Only Transformer (similar to modern LLMs like LLaMA / nanoGPT).
* **Positional Encoding:** Rotary Position Embedding (RoPE) or Learnable Positional Embeddings.
* **Normalization:** RMSNorm / LayerNorm with pre-normalization for training stability.
* **Attention Mechanism:** Scaled Dot-Product Multi-Head Self-Attention with causal masking.
* **Activation Function:** SwiGLU or GeLU (Gaussian Error Linear Unit).
* **Loss Function:** Cross-Entropy Loss with label smoothing option.
* **Optimization:** AdamW optimizer with cosine learning rate schedule and linear warmup.

### 2.2 Tech Stack
| Component | Tool / Library | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core language |
| **Mathematics & Tensors** | NumPy / PyTorch | Numerical computation & tensor acceleration |
| **Tokenization** | Custom Byte-Pair Encoding (BPE) / Character Tokenizer | Text processing from raw strings to IDs |
| **Configuration** | Pydantic / PyYAML | Strongly-typed model and training configs |
| **Testing** | `pytest`, `pytest-cov` | Unit tests for tensor shapes, gradients, and layers |
| **Code Quality** | `ruff`, `black`, `mypy` | Linting, formatting, and static type checking |
| **Serving & Interface** | `FastAPI`, `Rich` (CLI) | Interactive terminal inference and RESTful API endpoint |
| **Experiment Tracking** | Built-in TensorBoard / JSON logging | Tracking loss, learning rate, and perplexity |

---

## 3. Directory Layout (Professional Package Structure)

```text
ParagAI/
├── .plans/
│   └── project_plan.md              # Master project plan
├── configs/
│   ├── model_small.yaml             # Small configuration (fast local prototyping)
│   └── train_config.yaml            # Hyperparameters, batch size, learning rates
├── data/
│   ├── raw/                         # Raw text corpora (e.g., TinyShakespeare, WikiText)
│   └── processed/                   # Tokenized binary/cached datasets
├── src/
│   └── paragai/
│       ├── __init__.py
│       ├── config.py                # Pydantic schemas for configs
│       ├── engine/                  # Tier 1: Micro-autograd & scalar engine
│       │   ├── __init__.py
│       │   ├── value.py             # Autograd scalar graph node
│       │   └── nn.py                # Linear, Neuron, MLP
│       ├── tokenizer/               # Custom tokenizer implementation
│       │   ├── __init__.py
│       │   ├── base.py              # Base Tokenizer abstract class
│       │   ├── char_tokenizer.py    # Clean character-level tokenizer
│       │   └── bpe_tokenizer.py     # Subword / BPE tokenizer
│       ├── model/                   # Tier 2: Transformer architecture
│       │   ├── __init__.py
│       │   ├── attention.py         # Causal Multi-Head Attention
│       │   ├── normalization.py     # RMSNorm / LayerNorm
│       │   ├── feed_forward.py      # MLP / SwiGLU layer
│       │   ├── transformer_block.py # Residual Transformer decoder block
│       │   └── transformer.py       # Full ParagAI language model
│       ├── training/
│       │   ├── __init__.py
│       │   ├── dataset.py           # Dataloader & batching
│       │   ├── trainer.py           # Training loop, validation, gradient clipping
│       │   ├── lr_scheduler.py      # Cosine annealing with warmup
│       │   └── checkpoint.py        # Model saving/resuming state dicts
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── sampler.py           # Top-k, Top-p (nucleus), temperature sampling
│       │   └── generate.py          # Autoregressive generation pipeline
│       └── utils/
│           ├── logger.py            # Rich logging & progress bars
│           └── seed.py              # Deterministic reproducibility utility
├── tests/
│   ├── test_autograd.py             # Gradient sanity check vs PyTorch
│   ├── test_tokenizer.py            # Encoding/decoding roundtrip tests
│   ├── test_attention.py            # Causal mask & output shape tests
│   └── test_model.py                # Forward pass & loss test
├── scripts/
│   ├── train.py                     # Entry point for training runs
│   ├── evaluate.py                  # Evaluation & perplexity calculation
│   ├── chat.py                      # Interactive CLI demo with streaming
│   └── serve.py                     # FastAPI REST server
├── pyproject.toml                   # Project metadata and dependencies
├── README.md                        # Documentation and architecture diagrams
└── Makefile                         # Easy development shortcuts (lint, test, run)
```

---

## 4. Phase-by-Phase Roadmap

### Phase 1: Environment Setup & Foundations
* **Objectives:**
  - Setup Python virtual environment and modern packaging (`pyproject.toml`).
  - Configure code formatters, linters (`ruff`), and testing harness (`pytest`).
  - Implement Tier 1 micro-autograd engine (`src/paragai/engine/value.py`) to verify backpropagation, automatic differentiation, and gradient descent mechanics.
* **Deliverable:** Working scalar autograd engine verified against numerical gradient checks.

### Phase 2: Tokenization & Data Processing Pipeline
* **Objectives:**
  - Implement a clean abstract tokenizer interface.
  - Build `CharTokenizer` (fast, deterministic, zero-dependency baseline).
  - Build custom Byte-Pair Encoding (`BPETokenizer`) algorithm with vocabulary save/load functionality.
  - Implement streaming/chunked dataset loader with training/validation splits.
* **Deliverable:** Tokenizer pipeline capable of training on raw text corpora, saving vocabulary, and emitting batch tensors `(B, T)`.

### Phase 3: Core Transformer Model Implementation
* **Objectives:**
  - Implement `RMSNorm` / `LayerNorm` with learnable scale parameters.
  - Implement `MultiHeadAttention` with causal triangular masking to prevent looking into the future.
  - Implement Feed-Forward Network (FFN) with expansion factor (4x) and modern activations (GeLU/SwiGLU).
  - Implement `TransformerBlock` combining pre-norm, attention, residual connections, and FFN.
  - Assemble full `ParagAIModel` with token embeddings, positional embeddings, stacked blocks, and final linear head tied to input embeddings.
* **Deliverable:** Unit tests verifying tensor shapes `(Batch, Sequence, VocabSize)` and zero memory leaks.

### Phase 4: Training Pipeline, Optimization & Checkpointing
* **Objectives:**
  - Implement AdamW optimizer handling weight decay decoupled from learning rate.
  - Implement Cosine Annealing learning rate schedule with warmup.
  - Create robust `Trainer` class featuring:
    - Gradient clipping to prevent exploding gradients.
    - Checkpoint manager (save latest, best validation loss, resume training).
    - Periodic validation and cross-entropy loss tracking.
* **Deliverable:** Training script executable via `python scripts/train.py --config configs/train_config.yaml`.

### Phase 5: Generation & Sampling Engine
* **Objectives:**
  - Implement autoregressive text generation loop.
  - Build probabilistic samplers:
    - Temperature scaling (controls randomness/creativity).
    - Top-$k$ filtering.
    - Top-$p$ (Nucleus) sampling.
    - Repetition penalty.
  - Build an interactive terminal chat interface using `Rich` for streaming tokens as they generate.
* **Deliverable:** Interactive CLI (`python scripts/chat.py --checkpoint checkpoints/best.pt`).

### Phase 6: Testing, Benchmarking & Evaluation
* **Objectives:**
  - Comprehensive unit test suite covering:
    - Tokenizer encoding/decoding consistency.
    - Gradient correctness.
    - Determinism across random seeds.
    - Model export/import fidelity.
  - Evaluation script calculating validation perplexity ($PPL = \exp(\text{CrossEntropyLoss})$).
* **Deliverable:** 90%+ test coverage and standardized evaluation report.

### Phase 7: Professional Packaging & API Serving
* **Objectives:**
  - FastAPI web server (`scripts/serve.py`) with `/v1/models` and `/v1/completions` endpoints following standard API conventions.
  - Dockerfile for reproducible containerized execution.
  - Clean `README.md` complete with architecture explanation, quickstart guide, and hyperparameter cheat sheet.
* **Deliverable:** Production-ready repository ready for open-source publication or portfolio showcase.

---

## 5. Model Configuration Specifications (Default Baseline)

```yaml
# configs/model_small.yaml
model:
  vocab_size: 2048         # Configurable based on tokenizer
  max_seq_len: 256         # Context window length (T)
  n_embd: 192              # Embedding dimension
  n_head: 6                # Number of attention heads (head_dim = 32)
  n_layer: 6               # Number of transformer blocks
  dropout: 0.1             # Dropout probability
  bias: false              # Linear layers without bias (modern LLM standard)

training:
  batch_size: 32           # Number of sequences per batch (B)
  learning_rate: 6.0e-4    # Peak learning rate
  min_lr: 6.0e-5          # Minimum learning rate after decay
  warmup_iters: 200        # Warmup steps
  max_iters: 5000          # Total training iterations
  weight_decay: 0.1        # AdamW weight decay
  grad_clip: 1.0           # Gradient norm clipping threshold
```

---

## 6. Immediate Next Steps

1. **Step 1:** Initialize repository environment, `pyproject.toml`, and `.gitignore`.
2. **Step 2:** Implement the foundational math engine (`engine/value.py`) to demonstrate autograd from scratch.
3. **Step 3:** Implement the tokenizer module and verify with a sample corpus.
4. **Step 4:** Build the core Transformer model layers step by step with unit tests.

