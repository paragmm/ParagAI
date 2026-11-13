# ParagAI Architecture & Codebase Guide

This document provides a comprehensive, file-by-file and folder-by-folder breakdown of the **ParagAI** repository. It details the underlying mathematical concepts, structural decisions, data flow pipelines, and individual component responsibilities.

---

## 1. High-Level Architectural Vision

ParagAI is built on two core foundational principles:
1. **First-Principles Understanding:** Avoiding black-box libraries (such as `transformers` or `langchain`) for the model architecture. Every component—from scalar autograd nodes and rotary embeddings to causal attention and subword tokenization—is implemented explicitly.
2. **Dual-Tier System Design:**
   - **Tier 1 (Micro-Autograd Engine):** A pure Python/NumPy scalar automatic differentiation engine (`src/paragai/engine/`) demonstrating reverse-mode autograd, computational graph building, and backpropagation from scratch.
   - **Tier 2 (High-Performance Transformer):** A PyTorch-accelerated, modern decoder-only architecture (`src/paragai/model/`) following state-of-the-art open-weights designs (LLaMA/Mistral style), featuring Rotary Positional Embeddings (RoPE), Root Mean Square Normalization (RMSNorm), SwiGLU activations, Scaled Dot-Product Attention (SDPA), and Key-Value (KV) Caching.

```
                    ┌──────────────────────────────────────────────┐
                    │               PARAGAI STACK                  │
                    └──────────────────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌─────────────────────────────────┐                         ┌─────────────────────────────────┐
│     Tier 1: Scalar Autograd     │                         │   Tier 2: Production Decoder    │
│    (src/paragai/engine/)        │                         │      (src/paragai/model/)       │
│  - Value (Scalar DAG Node)      │                         │  - RMSNorm (Pre-normalization)  │
│  - Reverse-mode autodiff        │                         │  - RoPE (Rotary Embeddings)     │
│  - Neuron, Layer, MLP           │                         │  - Causal Self-Attention + SDPA │
│  - Pure Python & NumPy          │                         │  - SwiGLU Feed-Forward Network  │
└─────────────────────────────────┘                         │  - KV-Caching for O(1) Decode   │
                                                            │  - Tied Token & Head Weights    │
                                                            └─────────────────────────────────┘
```

---

## 2. Directory Tree Overview

```text
ParagAI/
├── .agents/                 # AI Developer Agent directives, workspace rules, and specialized skills
│   ├── skills/              # Specialized domain knowledge (model, trainer, tokenizer, ollama)
│   └── AGENTS.md            # Repository development guidelines, coding standards, and invariants
├── .plans/                  # Strategic roadmaps, Bengali & English architecture transition plans
├── configs/                 # Strongly-typed YAML configuration files for models and training
├── data/                    # Dataset storage
│   ├── raw/                 # Multilingual (Bengali/English), Chat, Automation, Technology raw text
│   └── processed/           # Merged corpora and pre-tokenized memory-mapped files
├── checkpoints/             # Serialized model weights (.pt, .safetensors, .gguf) and tokenizers
├── deploy/                  # Production deployment artifacts (Ollama Modelfile)
├── document/                # Guides for local deployment, Ollama generation, and benchmarking
├── notebooks/               # Interactive experimentation and Google Colab GPU training notebooks
├── scripts/                 # CLI entry points for data prep, tokenizer training, train, chat, export
├── src/
│   └── paragai/             # Core library package
│       ├── config.py        # Pydantic configuration schemas
│       ├── engine/          # Tier 1 scalar computational graph & autograd
│       ├── model/           # Tier 2 Transformer neural network layers
│       ├── tokenizer/       # BPE and Character-level tokenizer implementations
│       ├── training/        # Training loop, dynamic batching dataset, and LR schedulers
│       ├── inference/       # Autoregressive generation, KV-caching, probabilistic samplers
│       └── utils/           # Deterministic seeding and structured logging
├── tests/                   # Pytest test suite enforcing shape, causal, and gradient invariants
├── pyproject.toml           # PEP 517/621 project configuration and dependencies
├── requirements.txt         # Pinned top-level dependencies
└── README.md                # Project landing documentation
```

---

## 3. Detailed Folder and File Breakdown

### 3.1. `src/paragai/` (Core Framework)

#### Configuration:
* **`src/paragai/config.py`**
  * **Role:** Defines strongly typed configuration dataclasses using Pydantic (`ModelConfig` and `TrainConfig`).
  * **Key Classes:**
    * `ModelConfig`: Specifies `vocab_size`, `max_seq_len`, `n_embd`, `n_head`, `n_layer`, `dropout`, `bias`, `rms_norm_eps`, and `rope_theta`. Includes validation properties like `head_dim = n_embd // n_head`.
    * `TrainConfig`: Controls `batch_size`, `learning_rate`, `min_lr`, `warmup_iters`, `max_iters`, `weight_decay`, `grad_clip`, `eval_interval`, `device`, and paths.

#### Tier 1: Micro-Autograd Engine (`src/paragai/engine/`):
* **`src/paragai/engine/value.py`**
  * **Role:** The atomic building block of Tier 1. Represents a scalar value inside a dynamically constructed computational directed acyclic graph (DAG).
  * **Core Mechanisms:** Overloads standard Python mathematical operators (`+`, `-`, `*`, `/`, `**`, `exp()`, `log()`, `relu()`). Stores references to child nodes (`_prev`), the operation (`_op`), accumulated gradient (`self.grad`), and a local closure (`_backward()`).
  * **Backpropagation:** `backward()` builds a topologically sorted list of nodes and traverses it in reverse order, applying the multivariable chain rule.
* **`src/paragai/engine/nn.py`**
  * **Role:** Neural network abstractions constructed strictly from `Value` objects.
  * **Key Classes:**
    * `Module`: Base class providing `zero_grad()` and parameter aggregation.
    * `Neuron`: Linear unit $y = \text{ReLU}\left(\sum w_i x_i + b\right)$ or linear activation.
    * `Layer`: Collection of independent parallel neurons.
    * `MLP`: Multi-Layer Perceptron composing sequential layers with non-linear activations.

#### Tier 2: Neural Model Architecture (`src/paragai/model/`):
* **`src/paragai/model/normalization.py`**
  * **Role:** Implements Root Mean Square Normalization (`RMSNorm`).
  * **Equation:**
    $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
  * **Rationale:** Modulates activations without computing the mean, reducing computational overhead and memory traffic compared to standard LayerNorm while maintaining training stability.
* **`src/paragai/model/rope.py`**
  * **Role:** Rotary Position Embeddings (RoPE).
  * **Core Functions:**
    * `precompute_rope_freqs(head_dim, max_seq_len, theta)`: Precomputes complex frequency vectors $\Theta = \theta^{-2(i-1)/d}$ on the complex unit circle:
      $$\text{cis}(m \theta) = \cos(m \theta) + i \sin(m \theta)$$
    * `apply_rope(x, freqs_cis)`: Reshapes real query/key vectors into 2D coordinate pairs, treats them as complex numbers, performs rotation via complex multiplication, and unpacks back to real floats.
* **`src/paragai/model/attention.py`**
  * **Role:** Causal Multi-Head Self-Attention (`CausalSelfAttention`).
  * **Features:**
    * Independent linear projections for Query, Key, Value, and Output (`q_proj`, `k_proj`, `v_proj`, `out_proj`).
    * RoPE integration on $Q$ and $K$.
    * **KV-Cache:** Accepts cached keys and values from preceding generation steps and concatenates new projections along the time dimension.
    * **PyTorch 2.0 SDPA:** Calls `torch.nn.functional.scaled_dot_product_attention`, automatically selecting FlashAttention-2 or memory-efficient kernels when executed on supported GPUs.
* **`src/paragai/model/feed_forward.py`**
  * **Role:** SwiGLU Feed-Forward Network (`SwiGLUFeedForward`).
  * **Equation:**
    $$\text{FFN}_{\text{SwiGLU}}(x) = \left(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\right) W_{\text{down}}$$
  * **Dimension Scaling:** Defaults to an intermediate hidden dimension of approximately $\frac{8}{3} d_{\text{model}}$, rounded to a clean multiple of 64 for optimal GPU memory alignment.
* **`src/paragai/model/transformer_block.py`**
  * **Role:** A single decoder transformer layer (`TransformerBlock`).
  * **Flow:** Pre-normalization architecture with residual skip connections:
    $$x_1 = x + \text{Attention}(\text{RMSNorm}_1(x))$$
    $$x_2 = x_1 + \text{SwiGLU}(\text{RMSNorm}_2(x_1))$$
* **`src/paragai/model/transformer.py`**
  * **Role:** Complete autoregressive language model (`ParagAIModel`).
  * **Components:** Token embedding layer, RoPE frequency buffer, stack of `TransformerBlock` layers, final `RMSNorm`, and un-embedding linear projection (`lm_head`).
  * **Weight Tying:** `self.token_embedding.weight = self.lm_head.weight`, drastically reducing parameter footprint and improving representation alignment.
  * **Inference / Training Modes:** Supports standard forward passes (returning logits and cross-entropy loss) or KV-cache passes (returning next-token logits and updated cache buffers).

#### Tokenization (`src/paragai/tokenizer/`):
* **`src/paragai/tokenizer/base.py`**
  * **Role:** Abstract base class (`BaseTokenizer`) enforcing contracts for `encode`, `decode`, `vocab_size`, `save`, and `load`, along with standard special token IDs (`<pad>`, `<unk>`, `<bos>`, `<eos>`).
* **`src/paragai/tokenizer/char_tokenizer.py`**
  * **Role:** Deterministic character-level tokenizer. Useful for lightweight debugging, micro-models, and testing edge cases.
* **`src/paragai/tokenizer/bpe_tokenizer.py`**
  * **Role:** Subword Byte-Pair Encoding (BPE) tokenizer implemented from scratch.
  * **Algorithm:** Initializes vocabulary with special tokens and all 256 individual byte values (ensuring lossless Unicode handling). Iteratively calculates most frequent adjacent byte pairs and merges them until target vocabulary size is reached. Encodes and decodes UTF-8 byte streams safely.

#### Training Infrastructure (`src/paragai/training/`):
* **`src/paragai/training/dataset.py`**
  * **Role:** In-memory, memory-efficient dataset loader (`TextDataset`).
  * **Batching:** `get_batch(split, batch_size, seq_len, device)` extracts random contiguous sequences of length `seq_len` as inputs $X$ and shifted targets $Y = X_{t+1}$.
* **`src/paragai/training/lr_scheduler.py`**
  * **Role:** Learning rate scheduler with linear warmup and cosine decay (`CosineWarmupScheduler`).
  * **Behavior:** Linearly scales from 0 to $\text{max\_lr}$ over `warmup_iters`, then decays following a half-cosine curve down to $\text{min\_lr}$.
* **`src/paragai/training/trainer.py`**
  * **Role:** Encapsulates the complete training loop (`Trainer`).
  * **Optimizations:** AdamW optimizer configured with decoupled weight decay (decay applied strictly to 2D matrix weights, excluding 1D normalization gains and biases). Includes gradient clipping (`grad_clip = 1.0`), periodic validation loss calculation, and automatic checkpoint persistence.

#### Inference & Generation (`src/paragai/inference/`):
* **`src/paragai/inference/sampler.py`**
  * **Role:** Probabilistic token sampling (`sample_next_token`).
  * **Features:**
    1. **Repetition Penalty:** Down-scales or divides logits of previously generated tokens.
    2. **Temperature Scaling:** Softens or sharpens the distribution.
    3. **Top-K Filtering:** Restricts candidate tokens to the $K$ highest-probability items.
    4. **Top-P (Nucleus) Filtering:** Dynamically restricts candidates to the smallest cumulative probability set exceeding threshold $P$.
* **`src/paragai/inference/generate.py`**
  * **Role:** Autoregressive generation manager (`Generator`).
  * **KV-Cache Acceleration:** Handles the "Prefill" stage for the full prompt, then advances one token at a time with $O(1)$ computational complexity per step, eliminating redundant attention re-computation. Also provides an asynchronous token-by-token stream generator (`generate_stream`).

#### Utilities (`src/paragai/utils/`):
* **`src/paragai/utils/seed.py`**
  * **Role:** Deterministic seed configuration across Python `random`, NumPy, and PyTorch (CPU and CUDA).
* **`src/paragai/utils/logger.py`**
  * **Role:** Structured terminal logging using the `rich` library.

---

### 3.2. `scripts/` (Execution Entry Points)

* **`scripts/prepare_data.py`**
  * Scans `data/raw/` across categories (`stories`, `technology`, `chat`, `automation`, `translation`), normalizes text, and compiles `data/processed/combined_corpus.txt`.
* **`scripts/build_sft_dataset.py`**
  * Constructs conversational Supervised Fine-Tuning (SFT) datasets formatted in standard ChatML notation (`<|im_start|>system...`, `<|im_start|>user...`, `<|im_start|>assistant...`).
* **`scripts/train_bpe.py`**
  * Reads raw training corpora and trains a BPE subword vocabulary up to the target vocabulary size (e.g., 2048 tokens), exporting `checkpoints/tokenizer_bpe.json`.
* **`scripts/train.py`**
  * Main training script. Loads YAML configs, instantiates `ParagAIModel`, initializes `TextDataset`, and runs the `Trainer` loop.
* **`scripts/chat.py`**
  * Rich terminal interactive chat interface. Loads model weights and tokenizer, offering a conversation loop with configurable temperature, top-p, top-k, and ChatML prompt wrapping.
* **`scripts/export_hf.py`**
  * Converts native `.pt` checkpoints into HuggingFace-compatible formats (`model.safetensors`, `config.json`).
* **`scripts/export_gguf.py`**
  * Converts model weights and BPE/Char tokenizers into binary **GGUF** format (`checkpoints/paragai.gguf`) for execution in **Ollama** or `llama.cpp`. Accurately maps SwiGLU parameters, RoPE theta, and GPT-2 style byte token encodings.

---

### 3.3. `deploy/` & `document/` (Deployment & Guides)

* **`deploy/Modelfile`**
  * Specification for Ollama deployment. Points to `checkpoints/paragai.gguf`, configures sampling hyperparameters, sets ChatML template rules, and registers stop tokens (`<|im_end|>`, `<|endoftext|>`).
* **`document/ollama_model_generation_guide.md`** (and Bengali version `_bn.md`)
  * Step-by-step instructions for converting models, building local Ollama models (`ollama create paragai -f deploy/Modelfile`), and verifying REST endpoints (`/api/generate`, `/api/chat`).

---

### 3.4. `configs/` (Hyperparameter Declarations)

* **`configs/model_small.yaml`:** Small prototype model (e.g., 6 layers, 192 embedding dim, 6 heads, context window 256).
* **`configs/model_medium.yaml`:** Scaled architecture for deeper pre-training and higher capacity.
* **`configs/train_config.yaml`:** Standard training hyperparameters (batch size 32, max iters 5000, lr 6e-4, warmup 200).
* **`configs/train_demo.yaml`:** Ultra-fast configuration for smoke tests and CI verification.

---

### 3.5. `tests/` (Verification & Invariant Suites)

* **`tests/test_autograd.py`:** Validates Tier 1 scalar autograd derivatives against analytical ground truths.
* **`tests/test_model.py`:** Verifies:
  1. Output tensor shapes across all components.
  2. RMSNorm scaling invariance.
  3. Causal masking integrity (ensuring future tokens never leak into earlier representations).
  4. Non-zero gradients across all trainable parameters.
* **`tests/test_tokenizer.py`:** Asserts lossless round-trip encoding/decoding (`decode(encode(text)) == text`) for both Character and BPE tokenizers.
* **`tests/test_generation.py`:** Tests autoregressive decoding, KV-cache consistency, and sampling constraints.
* **`tests/test_trainer.py`:** Tests loss reduction over training steps and checkpoint serialization/deserialization.
* **`tests/test_export.py`:** Validates SafeTensors and GGUF serialization pipelines.

---

## 4. End-to-End Operational Workflows

### 4.1. Training Pipeline
```
Raw Text Corpora (data/raw/)
       │
       ▼
scripts/prepare_data.py ──► scripts/train_bpe.py ──► checkpoints/tokenizer_bpe.json
                                                           │
                                                           ▼
configs/model_small.yaml + configs/train_config.yaml ──► scripts/train.py
                                                           │
                                                           ▼
                                                checkpoints/best_model.pt
```

### 4.2. Inference Pipeline (KV-Cached)
```
Input Prompt: "ParagAI is"
       │
       ▼
Tokenize (encode) ──► [Token IDs]
                         │
                         ▼
Prefill Step: Compute Initial Attention & populate KV-Cache
                         │
                         ▼
Loop Step:
  1. Forward Pass single token using past (K, V)
  2. Sample Next Token (Temperature, Top-K, Top-P)
  3. Append to Generation buffer
  4. Repeat until EOS or Max Tokens reached
```

### 4.3. Ollama Export Pipeline
```
checkpoints/best_model.pt
       │
       ▼
scripts/export_hf.py ──► checkpoints/hf/model.safetensors
                                  │
                                  ▼
scripts/export_gguf.py ──► checkpoints/paragai.gguf
                                  │
                                  ▼
deploy/Modelfile ──► `ollama create paragai -f deploy/Modelfile` ──► `ollama run paragai`
```

---

## 5. Architectural Invariants Summary

| Feature | Implementation | Purpose & Advantage |
| :--- | :--- | :--- |
| **Autograd** | Pure Python DAG (`Value`) | First-principles conceptual clarity and validation |
| **Attention** | Causal SDPA with KV-Cache | $O(1)$ decoding per token; FlashAttention hardware speed |
| **Positional Encoding** | Rotary Embeddings (RoPE) | Relative position awareness without absolute position embeddings |
| **Normalization** | Pre-RMSNorm | 20-30% faster than LayerNorm; stabilizes deep gradient flow |
| **Feed-Forward** | SwiGLU | Superior token representation over standard ReLU/GELU |
| **Weight Sharing** | Tied Embedding & LM Head | Halves embedding parameter count; improves vocabulary alignment |
| **Deployment** | GGUF + Ollama | Run locally with zero external Python dependencies on CPU/GPU |
