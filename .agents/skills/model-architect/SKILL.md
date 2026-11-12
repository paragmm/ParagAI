---
name: model-architect
description: >-
  Use this skill when designing, implementing, modifying, or debugging the neural network layers,
  mathematical foundations, attention mechanisms, or Transformer architecture of ParagAI.
---

# ParagAI Model Architect Skill

This skill guides the design, implementation, and verification of the neural network layers in `ParagAI`. It covers both the Tier-1 scalar autograd engine and Tier-2 Decoder-Only Transformer.

## Architecture Specification

ParagAI adheres to the modern Decoder-Only Transformer standard with Pro-Grade enterprise optimizations:
* **Position Encoding:** Rotary Position Embedding (RoPE) with dynamic offset slicing for cached generation
* **Normalization:** RMSNorm with learnable scaling $\gamma$, no bias
* **Attention Engine:** PyTorch 2.0 SDPA (`F.scaled_dot_product_attention`) with native C++ FlashAttention kernel acceleration
* **Inference Engine:** Key-Value (KV) Cache with $O(1)$ constant time step generation
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
- Computes complex 2D rotation on $(Q, K)$ heads before dot-product attention.
- Frequency base default: $\theta = 10000.0$.
- Preserves relative token distance decaying over long sequences.
- Offset calculation for KV-Cache:
  ```python
  past_len = kv_cache[0].size(2) if kv_cache is not None else 0
  freqs = freqs_cis[past_len : past_len + T]
  ```

### 3. Pro-Grade Causal Self-Attention with KV-Cache & SDPA
- Location: `src/paragai/model/attention.py`
- Projections: $Q = X W_q$, $K = X W_k$, $V = X W_v$
- **KV-Cache Ingestion:** Concatenates past cached keys and values with current step:
  $$K_{\text{new}} = [K_{\text{past}} \,;\, K_{\text{current}}], \quad V_{\text{new}} = [V_{\text{past}} \,;\, V_{\text{current}}]$$
- **Hardware Acceleration:** PyTorch 2.0 `F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)` avoids quadratic $O(T^2)$ memory allocation and executes directly via optimized C++ FlashAttention kernels.

### 4. SwiGLU Feed-Forward Network (`FeedForward`)
- Location: `src/paragai/model/feed_forward.py`
  $$\text{SwiGLU}(x) = (x W_{\text{gate}} \cdot \text{SiLU}(x W_{\text{gate}})) \odot (x W_{\text{up}}) W_{\text{down}}$$
- Expansion factor: $\frac{8}{3} d_{\text{model}}$ (LLaMA-style).

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

