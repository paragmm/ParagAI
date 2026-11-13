# ParagAI: Large-Scale LLM Architectural Transformation Plan

> **Document:** Comprehensive System Architecture Guide & Scaling Roadmap  
> **Target:** Scaling ParagAI from Prototype (10M–100M) to Foundation LLM (3B, 7B, 13B, 70B+)  
> **File Location:** `.plans/07_large_scale_llm_transformation_plan.md`  
> **Status:** Planned Architectural Specification (Code Unchanged)  

---

## 1. Complete Project Structure & Purpose Matrix

The table below maps every directory and file in the ParagAI codebase, explaining its current role and whether changes are required for large-scale operation:

| Path | Current Purpose | Status for Large Scale | Required Transformation |
| :--- | :--- | :--- | :--- |
| **`src/paragai/config.py`** | Pydantic schema for `ModelConfig` and `TrainConfig`. | ⚠️ Needs Upgrade | Add `n_kv_head` (GQA), `rope_scaling` (YaRN/NTK), `gradient_accumulation_steps`, `mixed_precision` (`bf16`/`fp8`), `activation_checkpointing`. |
| **`src/paragai/model/attention.py`** | Causal Self-Attention with MHA, RoPE, SDPA, and KV-cache. | ⚠️ Critical Redesign | Implement **Grouped-Query Attention (GQA)**. Replace dynamic `torch.cat` KV-cache with pre-allocated circular buffers or PagedAttention. |
| **`src/paragai/model/transformer.py`** | Top-level `ParagAIModel` with tied weights & stacked blocks. | ⚠️ Needs Upgrade | Add `torch.utils.checkpoint` (Activation Checkpointing), untie input/output embeddings (`lm_head.weight` $\neq$ `token_embedding.weight`), support Tensor Parallel projections. |
| **`src/paragai/model/transformer_block.py`** | Pre-norm Transformer decoder block (Attention + SwiGLU). | ⚠️ Minor Upgrade | Integrate gradient checkpointing hook and parallel forward branches. |
| **`src/paragai/model/rope.py`** | Rotary Position Embeddings (RoPE) precomputation & rotation. | ⚠️ Needs Upgrade | Implement **YaRN / Dynamic NTK-aware RoPE** frequency interpolation for long contexts (32K–128K). |
| **`src/paragai/model/feed_forward.py`** | SwiGLU gated MLP (`w1`, `w2`, `w3`). | ✅ Scalable | Architecture is already LLaMA-3 compliant. (Optionally add Sparse Mixture-of-Experts / MoE gating for MoE models). |
| **`src/paragai/model/normalization.py`** | RMSNorm layer for numerical stabilization. | ✅ Scalable | Already production-grade and identical to LLaMA-3 / Gemma implementations. |
| **`src/paragai/training/trainer.py`** | Single-GPU training loop with AdamW and cosine schedule. | ⚠️ Critical Redesign | Integrate **Distributed Data Parallel (FSDP / DeepSpeed ZeRO-3)**, **Automatic Mixed Precision (`bf16`)**, and **Gradient Accumulation**. |
| **`src/paragai/training/dataset.py`** | In-memory token slice dataset. | ⚠️ Critical Redesign | Replace with **Memory-Mapped (`numpy.memmap`) Binary Streaming Dataset** with packed sequence chunking and document boundary masks. |
| **`src/paragai/training/lr_scheduler.py`** | Cosine decay with linear warmup. | ✅ Scalable | Core math is solid. Scale `max_iters` and tokens to billions. |
| **`src/paragai/tokenizer/bpe_tokenizer.py`** | Pure Python BPE tokenizer (~2,048 vocab). | ⚠️ Needs Replacement | Replace with **Rust-accelerated Tokenizer** (HuggingFace `tokenizers` / SentencePiece) with 32k–128k vocabulary and byte-fallback. |
| **`src/paragai/tokenizer/char_tokenizer.py`** | Character-level tokenizer. | ℹ️ Educational Only | Keep for unit tests / educational tier; unsuitable for large-scale LLMs. |
| **`src/paragai/tokenizer/base.py`** | Base class for tokenizers with special token constants. | ⚠️ Minor Upgrade | Expand special tokens for multi-turn ChatML, tool calling tags (`<\|tool_call\|>`), and fill-in-the-middle (FIM). |
| **`src/paragai/inference/generate.py`** | Autoregressive generator with KV-cache. | ⚠️ Needs Upgrade | Integrate continuous batching, pre-allocated static KV-cache tensors, and speculative decoding. |
| **`src/paragai/inference/sampler.py`** | Temperature, top-k, top-p nucleus, and repetition penalty. | ✅ Scalable | Mathematically complete; add Min-P and Mirostat sampling for cutting-edge generation. |
| **`src/paragai/engine/value.py` & `nn.py`** | Tier-1 scalar autograd engine (micrograd style). | ℹ️ Educational Only | Retain strictly as an educational tier; not used in GPU-accelerated LLM execution. |
| **`src/paragai/utils/seed.py` & `logger.py`** | Reproducibility and logging utilities. | ✅ Scalable | Add multi-rank rank-zero logging guards (`rank == 0`) for multi-GPU training. |
| **`scripts/train.py`** | Main entrypoint for model training. | ⚠️ Needs Upgrade | Add CLI flags for `accelerate launch` / `torchrun` multi-GPU execution and distributed configs. |
| **`scripts/export_gguf.py`** | PyTorch to GGUF exporter via `gguf-py`. | ✅ Scalable | Fully functional for local quantization (q4_k_m, q8_0) and Ollama edge execution. |
| **`scripts/export_hf.py`** | HuggingFace Hub exporter. | ✅ Scalable | Functional for uploading checkpoints to HuggingFace. |
| **`scripts/chat.py`** | Interactive terminal chat script. | ✅ Scalable | Add streaming token output (`yield` tokens in real-time). |
| **`configs/model_small.yaml` & `model_medium.yaml`** | Architecture hyperparameters. | ⚠️ Needs Scaling | Add configs for 1B, 3B, and 7B variants (e.g., `model_7b.yaml` with `n_embd: 4096`, `n_layer: 32`, `n_head: 32`, `n_kv_head: 8`). |
| **`deploy/Modelfile`** | Ollama definition file. | ✅ Scalable | Ready for Ollama deployment. |
| **`tests/`** | 21 Unit tests (autograd, model, tokenizer, trainer, export). | ⚠️ Needs Expansion | Add tests for GQA tensor shapes, gradient checkpointing invariance, and distributed rank synchronization. |

---

## 2. High-Priority Architectural Redesign Areas

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LARGE-SCALE LLM ARCHITECTURE ROADMAP                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ├─── 1. ATTENTION LAYER: Grouped-Query Attention (GQA)
         │       • n_head (e.g., 32) Queries vs n_kv_head (e.g., 8) Keys/Values
         │       • 75% reduction in KV-cache VRAM footprint
         │
         ├─── 2. MEMORY EFFICIENCY: Activation Checkpointing & Untied Weights
         │       • torch.utils.checkpoint to prevent activation memory blowup
         │       • Untied input/output embeddings to avoid gradient interference
         │
         ├─── 3. DISTRIBUTED ENGINE: FSDP / DeepSpeed ZeRO-3 & Tensor Parallelism
         │       • Multi-GPU gradient sharding, optimizer state partitioning
         │       • Automatic Mixed Precision (AMP) with bfloat16 / fp8
         │       • Gradient accumulation across micro-batches
         │
         ├─── 4. HIGH-THROUGHPUT DATA PIPELINE: Streaming Memory-Mapped Tokens
         │       • Fast binary datasets (uint16/uint32) via numpy.memmap
         │       • High-speed Rust-based BPE tokenizer (32k–128k vocabulary)
         │
         └─── 5. LONG-CONTEXT EXTENSION: Dynamic YaRN / NTK RoPE
                 • Expand context length from 256 to 32,768+ tokens without loss
```

---

## 3. Detailed Technical Blueprint for Modifications

### 3.1 Step 1: Upgrading Attention to GQA (`src/paragai/model/attention.py`)
* **Mathematical Shift:**
  $$\text{Query: } (B, T, n_{\text{head}}, d_k) \quad \text{vs} \quad \text{Key, Value: } (B, T, n_{\text{kv\_head}}, d_k)$$
  Where $n_{\text{kv\_head}} \ll n_{\text{head}}$ (typically $1:4$ ratio).
* Keys and Values are repeated/broadcasted across their corresponding query group during Scaled Dot-Product Attention:
  ```python
  # Repeat KV heads to match Q heads for SDPA
  k = torch.repeat_interleave(k, repeats=self.num_queries_per_kv, dim=1)
  v = torch.repeat_interleave(v, repeats=self.num_queries_per_kv, dim=1)
  ```

### 3.2 Step 2: Activation Checkpointing (`src/paragai/model/transformer.py`)
* During large-scale training, wrap each `TransformerBlock` forward execution:
  ```python
  if self.training and self.config.use_gradient_checkpointing:
      x, next_cache = torch.utils.checkpoint.checkpoint(
          block, x, self.freqs_cis, layer_cache, use_cache, use_reentrant=False
      )
  else:
      x, next_cache = block(x, freqs_cis=self.freqs_cis, kv_cache=layer_cache, use_cache=use_cache)
  ```

### 3.3 Step 3: Mixed Precision & Gradient Accumulation (`src/paragai/training/trainer.py`)
* Replace single-step FP32 backward passes with AMP and micro-batch accumulation:
  ```python
  with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
      _, loss = self.model(x, targets=y)
      loss = loss / self.config.gradient_accumulation_steps

  loss.backward()

  if (step + 1) % self.config.gradient_accumulation_steps == 0:
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
      self.optimizer.step()
      self.scheduler.step()
      self.optimizer.zero_grad(set_to_none=True)
  ```

### 3.4 Step 4: Long-Context YaRN RoPE Scaling (`src/paragai/model/rope.py`)
* Interpolate RoPE frequencies across high frequencies (local attention) and extrapolate across low frequencies (global context) to scale context windows from 256 to 32k tokens.

### 3.5 Step 5: High-Performance Data Loader (`src/paragai/training/dataset.py`)
* Implement zero-copy `numpy.memmap` reading pre-tokenized binary files (`train.bin`, `val.bin`), eliminating RAM bottlenecks for multi-billion token training runs.

---

## 4. Execution Readiness Summary

* **Current Code State:** Completely intact and working (21/21 unit tests passing).
* **Next Implementation Steps (When Ready to Code):**
  1. Add GQA parameters to `ModelConfig` and update `CausalSelfAttention`.
  2. Implement gradient checkpointing in `ParagAIModel`.
  3. Integrate `torchrun` and `torch.bfloat16` in `Trainer`.
  4. Train a 32K vocabulary BPE tokenizer with HuggingFace Tokenizers backend.
