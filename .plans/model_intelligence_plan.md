# ParagAI: Roadmap to Advanced Model Intelligence

> **Document:** Strategic Roadmap to Scale ParagAI from a Toy Model to an Intelligent AI Assistant  
> **Location:** `.plans/model_intelligence_plan.md`  
> **Target:** Fluency, reasoning, instruction-following, and bilingual competence (English + Bengali)  
> **Status:** Active Execution Plan  

---

## 1. Executive Summary: Why the Current Model is Limited

Our baseline prototype proved that the math, attention mechanism, and training loop work from first principles. However, the model currently has two fundamental bottlenecks:

1. **Data Starvation:** The current model was trained on **4,142 characters (~500 words)**. For comparison, ChatGPT was trained on over **10,000,000,000,000 words**. An AI cannot reason about concepts it has never read.
2. **Character-Level Fragmentation:** Using a character tokenizer forces the model to assemble letters one by one (`p-r-o-g-r-a-m-m-i-n-g`), wasting attention capacity on basic spelling rather than semantic concepts and logic.

To make **ParagAI** genuinely intelligent, fluent, and capable of real conversation, we follow the **5 Pillars of Modern AI Scaling**:

```text
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│ Pillar 1: Subword BPE  │ ───► │ Pillar 2: Dataset      │ ───► │ Pillar 3: Architecture │
│ Tokenizer (4k-8k Vocab)│      │ Scaling (10MB - 100MB) │      │ Expansion (15M-45M)    │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
            │                                                                │
            ▼                                                                ▼
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│ Pillar 6: Ollama & Web │ ◄─── │ Pillar 5: KV-Cache &   │ ◄─── │ Pillar 4: Instruction  │
│ UI Deployment (GGUF)   │      │ FlashAttention (Fast)  │      │ Tuning (Chat Template) │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

---

## 2. Pillar 1: Transition to Subword BPE Tokenizer

### The Problem with Character Tokenization:
* A 50-word sentence consumes ~300 character tokens, exceeding the small context window ($T=256$).
* The neural network spends 80% of its capacity learning English and Bengali phonetics.

### The Upgrade:
* Activate the already-implemented **`BPETokenizer`** (`src/paragai/tokenizer/bpe_tokenizer.py`).
* Target Vocabulary Size: **4,096 to 8,192 subwords**.
* *Impact:* Whole words (`intelligent`, `computer`, `কম্পিউটার`, `বাংলাদেশ`) become single integer tokens. The effective context length quadruples, and spelling errors drop to zero.

---

## 3. Pillar 2: Dataset Scaling (The #1 Determinant of Intelligence)

Modern research (Chinchilla Scaling Laws) proves that **data quality and token volume dictate 80% of model intelligence**.

### Recommended Staged Datasets:

| Stage | Dataset Name | Size | Training Time (Colab T4) | Capability Gained |
|---|---|---|---|---|
| **Level 1** | **TinyStories + Bengali Books** | 10 MB (~2M words) | 15–20 minutes | Coherent storytelling, perfect grammar, full sentences |
| **Level 2** | **Alpaca Cleaned + Bangla-Alpaca** | 25 MB (~5M words) | 30–45 minutes | Direct Q&A, instruction-following, task execution |
| **Level 3** | **OpenWebText Subset / WikiText** | 100 MB (~20M words) | 1.5–2 hours | General world knowledge, facts, science, and coding |

### High-Quality Open Datasets Ready for Direct Download:
* **TinyStories (`roneneldan/TinyStories`):** Synthetic English stories produced by GPT-4 designed specifically to teach models under 50M parameters how to reason.
* **Bangla-Alpaca (`shahruk10/bangla-alpaca`):** 52,000 instruction-response pairs in Bengali.
* **Bengali Wikipedia Cleaned:** High-quality non-toxic factual knowledge in Bengali.

---

## 4. Pillar 3: Supervised Instruction Fine-Tuning (SFT)

Currently, the model only does **raw text continuation** (like completing autocomplete sentences). To make it a real assistant like ChatGPT, we format all training data into standard **ChatML** instruction pairs:

```text
<|im_start|>system
You are ParagAI, a helpful, intelligent AI assistant.<|im_end|>
<|im_start|>user
What is photosynthesis?<|im_end|>
<|im_start|>assistant
Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water.<|im_end|>
```

When trained on 10,000+ such pairs, the model learns:
1. To never mimic the user.
2. To provide helpful, structured answers.
3. To stop speaking immediately when emitting `<|im_end|>`.

---

## 5. Pillar 4: Model Architecture Scaling

Our current architecture has ~2.6M parameters. We can safely scale it up to **15M to 45M parameters**, which still trains comfortably on a free Google Colab T4 GPU (16 GB VRAM):

### Proposed Model Configuration Specs:

```yaml
# configs/model_medium.yaml
model:
  name: "ParagAI-Medium"
  vocab_size: 4096         # Subword BPE vocabulary
  max_seq_len: 512         # Doubled context window length
  n_embd: 384              # 2x wider representation vector
  n_head: 8                # 8 attention heads (head_dim = 48)
  n_layer: 8               # 8 transformer blocks
  dropout: 0.1
  bias: false
  rms_norm_eps: 1.0e-5
  rope_theta: 10000.0
# Total Parameters: ~18.5 Million (Trains in 15 mins on Colab T4)
```

---

## 6. Pillar 5: Inference Acceleration (KV-Cache)

Currently, for every new token generated, the model recomputes self-attention across all previous tokens from scratch ($O(T^2)$ computational complexity).

* **The Upgrade:** Implement a **Key-Value (KV) Cache** in `CausalSelfAttention`.
* **How it works:** Cache the $(K, V)$ projections of past tokens so generation only computes the single newest query vector ($O(1)$ per step).
* **Speedup:** Generation becomes **10x to 30x faster**, allowing instant streaming of words without latency.

---

## 7. Actionable Implementation Schedule

### Step 1 (Immediate - 30 mins):
* Download a real 10 MB dataset (TinyStories + Bengali Wikipedia subset).
* Train the BPE tokenizer to 4,096 vocabulary size.
* Re-train on Colab with `configs/model_medium.yaml` for 15 minutes.
* *Result: Immediate jump from broken sentences to fluent, grammatically sound paragraphs.*

### Step 2 (Instruction Tuning):
* Convert 5,000 instruction pairs into ChatML format.
* Train for 10 epochs.
* *Result: Model answers questions directly as an assistant.*

### Step 3 (Deploy & Quantize):
* Export to GGUF format (`Q4_K_M`).
* Import into Ollama and connect to **Open WebUI** for a ChatGPT-like desktop experience.

