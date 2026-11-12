---
name: model-architect
description: >-
  Use this skill when designing, implementing, modifying, or debugging the neural network layers,
  mathematical foundations, attention mechanisms, or Transformer architecture of ParagAI.
---

# ParagAI Model Architect Skill

This skill guides the design, implementation, and verification of the neural network layers in `ParagAI`. It covers both the Tier-1 scalar autograd engine and Tier-2 Decoder-Only Transformer.

## Architecture Specification

ParagAI adheres to the modern Decoder-Only Transformer standard:
* **Position Encoding:** Rotary Position Embedding (RoPE)
* **Normalization:** RMSNorm with learnable scaling $\gamma$, no bias
* **Attention:** Scaled Dot-Product Attention with triangular causal mask
* **Feed-Forward:** SwiGLU (Gated Linear Unit with Swish/SiLU activation)
* **Tied Embeddings:** Output projection layer shares weights with token input embeddings

---

## Component Implementation Checklist

### 1. Root Mean Square Normalization (`RMSNorm`)
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
- Location: `src/paragai/model/normalization.py`
- Verification: Mean squared values should normalize to approximately 1.0; output shape matches input `(B, T, C)`.

### 2. Rotary Position Embedding (`RoPE`)
- Location: `src/paragai/model/rope.py`
- Computes complex or 2D rotation on $(Q, K)$ heads before dot-product attention.
- Frequency base default: $\theta = 10000.0$.
- Preserves relative token distance decaying over long sequences.

### 3. Causal Multi-Head Self-Attention (`MultiHeadAttention`)
- Location: `src/paragai/model/attention.py`
- Projections: $Q = X W_q$, $K = X W_k$, $V = X W_v$
- Causal Mask: Upper-triangular matrix filled with $-\infty$ (or $-10000.0$) where $j > i$ to prevent looking ahead.
- Attention Calculation:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
- Shape Invariant: Input `(B, T, C)` $\to$ Output `(B, T, C)`.

### 4. SwiGLU Feed-Forward Network (`FeedForward`)
- Location: `src/paragai/model/feed_forward.py`
  $$\text{SwiGLU}(x) = (x W_{\text{gate}} \cdot \text{SiLU}(x W_{\text{gate}})) \odot (x W_{\text{up}}) W_{\text{down}}$$
- Expansion factor: $4 \times d_{\text{model}}$ (or $\frac{8}{3} d_{\text{model}}$ for LLaMA-style).

---

## Validation & Test Procedures

When making changes to the architecture:
1. **Run Shape Validation:**
   ```bash
   pytest tests/test_attention.py -v
   pytest tests/test_model.py -v
   ```
2. **Check Gradient Flow:**
   Ensure no isolated tensors detach gradients (`requires_grad=True` on all parameter leaves).
3. **Verify Zero Memory Leak:**
   Confirm attention tensor caching doesn't retain unneeded graph histories during evaluation (`torch.no_grad()`).

