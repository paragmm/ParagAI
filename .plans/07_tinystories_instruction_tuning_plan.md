# ParagAI Step-by-Step TinyStories Instruction-Tuning Master Plan

**Document ID:** `PLAN-07`  
**Target Component:** Data Engineering, Automated Instruction Synthesis, Continual SFT Training & Ollama Deployment  
**Applicability:** `ParagAI-Small` (1.5M - 2.5M) and `ParagAI-Medium` (10M - 25M)  
**Status:** Approved for Execution  

---

## 1. Executive Summary & Core Objective

### 1.1 The Challenge
Standard language modeling on raw narrative text (such as raw stories) produces a **Base Completion Model**. When prompted with an open instruction like:
```text
write a story for me
```
A base model treats the prompt as an incomplete sentence prefix rather than a directive, generating outputs such as:
```text
...and take it to the teacher before the school bell rings.
```

### 1.2 The Solution: Instruction Tuning via ChatML
To transform ParagAI from a passive sentence-completer into an **autonomous directive-following assistant**, raw stories from Microsoft Research's `roneneldan/TinyStories` corpus will be programmatically converted into **Supervised Fine-Tuning (SFT) Instruction-Response pairs** using standard ChatML notation:
```text
<|im_start|>system
You are ParagAI, an imaginative, helpful, and creative storytelling assistant.<|im_end|>
<|im_start|>user
Write a creative story for me.<|im_end|>
<|im_start|>assistant
Once upon a time, in a bright green meadow bordered by tall whispering pines...<|im_end|>
```

### 1.3 Key Constraints & Architectural Guardrails
1. **Incremental Daily Training (Continual Learning):** Training will occur in controllable slices (5 MB to 10 MB per day) without requiring 100 MB to be processed simultaneously.
2. **Tokenizer Vocabulary Freezing:** The BPE tokenizer (`checkpoints/tokenizer_bpe.json` or `tokenizer.json`) remains strictly frozen to avoid PyTorch embedding tensor dimension mismatches.
3. **Catastrophic Forgetting Mitigation:** Replay buffers (mixing 10% prior chat, translation, and tool automation data into daily batches) ensure previous capabilities are retained.

---

## 2. High-Level Architecture & End-to-End Pipeline

```
┌────────────────────────────────────────────────────────┐
│  Phase 1: Data Acquisition                             │
│  Hugging Face: roneneldan/TinyStories                  │
│  Streaming download into daily chunks (5MB - 10MB)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Phase 2: Automated Instruction Synthesis              │
│  scripts/prepare_tinystories_sft.py                    │
│  Raw Story ──> Dynamic User Prompts ──> ChatML SFT     │
│  • Story Generation Directives                         │
│  • Entity/Character Specific Prompts                   │
│  • Moral / Lesson Queries                              │
│  • Bilingual (English & Bengali) Directives            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Phase 3: Sequential Supervised Fine-Tuning (SFT)     │
│  scripts/train.py --resume checkpoints/best_model.pt   │
│  • Controlled Learning Rate (1e-4 down to 5e-5)        │
│  • 1,500 - 3,000 steps per daily slice                 │
│  • Checkpoint preservation                             │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Phase 4: GGUF Export & Ollama Deployment              │
│  scripts/export_hf.py ──> scripts/export_gguf.py       │
│  ollama create paragai -f deploy/Modelfile             │
│  Verification: "write a story for me" ──> Full Story   │
└────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Implementation Roadmap

### Step 01: Sliced Data Acquisition from Hugging Face
Rather than downloading the full 1 GB TinyStories dataset, download sliced partitions using a streaming Python utility.

* **File Location:** `scripts/fetch_tinystories.py`
* **Target Storage:** `data/raw/stories/tinystories_chunk_01.txt`
* **Target Size:** ~5 MB to 10 MB (~10,000 to 20,000 clean stories).
* **Format:** Raw UTF-8 text separated by double newlines or end-of-text delimiters.

#### Acquisition Logic:
```python
import os
import urllib.request
import json

def fetch_tinystories_slice(output_path: str, target_size_mb: float = 10.0):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    target_bytes = int(target_size_mb * 1024 * 1024)
    # Stream from Hugging Face TinyStories raw parquet or text export
    # Write cleanly until target size is reached
```

---

### Step 02: Automated Instruction-Tuning Transformation
To teach the model how to follow commands like *"write a story for me"*, a transformation script reads raw stories and wraps them in diverse user instructions.

* **File Location:** `scripts/prepare_tinystories_sft.py`
* **Output Path:** `data/processed/stories_sft_chunk_01.txt`

#### Prompt Synthesizer Categories:
1. **General Story Request:**
   - *"Write a story for me."*
   - *"Please write an interesting and imaginative story."*
   - *"Can you tell me a story?"*
   - *"Compose a short story for me."*
2. **Character & Setting Extraction (Heuristic-based):**
   - Extract first named entity or animal (e.g., *"Leo the fox"*, *"Lily"*).
   - Synthesize prompt: *"Write a children's story about Lily and a magic garden."*
3. **Bedtime / Calming Directives:**
   - *"Tell me a calming bedtime story."*
   - *"Write a sweet story suitable for bedtime reading."*
4. **Bilingual / Bengali Directives:**
   - *"আমাকে একটি ছোট রূপকথার গল্প বলো।"*
   - *"আমার জন্য একটি সুন্দর ইংরেজি গল্প লিখো।"*
   - *"একটি শিক্ষণীয় রূপকথা লিখে শোনাও।"*

#### ChatML Formatting Structure:
```text
<|im_start|>system
You are ParagAI, a creative and imaginative AI storytelling assistant.<|im_end|>
<|im_start|>user
Write a story for me.<|im_end|>
<|im_start|>assistant
Once upon a time, there was a little bird named Pip. Pip had bright blue wings...<|im_end|>
```

---

### Step 03: Tokenizer Stability & Verification
* **Rule:** Never re-train or modify `tokenizer.json` or `tokenizer_bpe.json` during incremental SFT.
* **Integrity Check:**
  Before training starts, verify that `vocab_size` in the training configuration exactly matches `model.tok_embeddings.weight.shape[0]` (e.g., 142 for Char or 1024 for BPE).
* **Byte-Fallback Safety:** Any unexpected punctuation or symbols in TinyStories will be resolved into their constituent UTF-8 byte tokens without modifying vocabulary tables.

---

### Step 04: Sequential SFT Training Protocol (Day-by-Day)

When training sequentially on daily chunks, apply **Continual Learning Hyperparameters**:

| Parameter | Initial Pre-training | Daily SFT (Day 1) | Daily SFT (Day 2+) |
| :--- | :--- | :--- | :--- |
| **Resume Flag** | `None` | `--resume checkpoints/best_model.pt` | `--resume checkpoints/best_model.pt` |
| **Max Iterations** | 5,000 | 2,500 - 3,000 | 2,000 - 2,500 |
| **Learning Rate** | $6.0 \times 10^{-4}$ | $2.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ |
| **Min Learning Rate** | $6.0 \times 10^{-5}$ | $2.0 \times 10^{-5}$ | $1.0 \times 10^{-5}$ |
| **Warmup Steps** | 200 | 50 | 50 |
| **Batch Size** | 32 | 32 | 32 |
| **Sequence Length** | 256 | 256 | 256 |

#### Execution Command:
```powershell
python scripts/train.py `
    --resume checkpoints/best_model.pt `
    --data data/processed/stories_sft_chunk_01.txt `
    --config configs/train_config.yaml
```

---

### Step 05: Catastrophic Forgetting Prevention (Replay Mix)
To ensure ParagAI does not forget general conversation, Bengali translation, and shell automation tool execution when immersed in story data:
* Each daily training corpus must contain a **Replay Mix**:
  - **85%** New TinyStories SFT data.
  - **5%** General Chat conversations (`conversations_sft.txt`).
  - **5%** Bengali translation pairs (`parallel_sft.txt`).
  - **5%** Tool automation commands (`tools_sft.txt`).
* The data preparation script automatically injects these replay samples into every daily chunk.

---

### Step 06: Verification & Interactive Testing

#### 1. Terminal Interactive Validation:
Run [scripts/chat.py](file:///c:/Users/hedpa/Documents/AI/ParagAI/scripts/chat.py) with the updated checkpoint:
```powershell
python scripts/chat.py
```
Test with multiple prompt variations:
- `write a story for me`
- `Tell me a creative story about a brave cat`
- `Write a bedtime story`
- `একটি ছোট গল্প বলো`

**Success Criteria:**
1. Output begins directly with a narrative paragraph (e.g. *"Once upon a time..."*).
2. The response does NOT append to the prompt as a partial phrase.
3. The generation cleanly terminates at `<|im_end|>` without infinite repetition.

---

### Step 07: GGUF Conversion & Ollama Production Deployment

After each successful training iteration:
1. **Export Hugging Face SafeTensors:**
   ```powershell
   python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output_dir checkpoints/export_hf
   ```
2. **Convert to GGUF (with GPT-2 Byte Unicode Mapping):**
   ```powershell
   python scripts/export_gguf.py --input_dir checkpoints/export_hf --tokenizer checkpoints/tokenizer.json --output checkpoints/paragai.gguf
   ```
3. **Register in Ollama:**
   ```powershell
   ollama create paragai -f deploy/Modelfile
   ```
4. **Live Ollama Story Test:**
   ```powershell
   ollama run paragai:latest "write a story for me"
   ```

---

## 4. Daily Training Schedule & Progress Tracker

| Day | Target Dataset Chunk | Size | Estimated Steps | Target Competency Acquired |
| :--- | :--- | :--- | :--- | :--- |
| **Day 1** | `tinystories_sft_chunk_01.txt` | 5 MB | 2,500 | Recognizes *"write a story for me"* and generates simple 2-character stories. |
| **Day 2** | `tinystories_sft_chunk_02.txt` | 5 MB | 2,500 | Adds descriptive adjectives, environmental settings, and dialogue. |
| **Day 3** | `tinystories_sft_chunk_03.txt` | 8 MB | 3,000 | Introduces problem/resolution narrative arcs (conflict $\to$ resolution). |
| **Day 4** | `bengali_stories_sft_chunk_01.txt`| 5 MB | 2,500 | Learns Bengali storytelling syntax (*"এক দেশে এক..."*). |
| **Day 5+**| Incremental 10 MB Batches | 10 MB | 3,000 | Multi-paragraph coherence and consistent character identity. |

---

## 5. Summary of Deliverables to Build

1. `scripts/fetch_tinystories.py`: Streaming downloader for Hugging Face TinyStories slices.
2. `scripts/prepare_tinystories_sft.py`: Programmatic ChatML instruction synthesizer with replay buffer.
3. `configs/sft_config.yaml`: Dedicated fine-tuning configuration with tuned learning rate ($2 \times 10^{-4}$).
4. `.plans/07_tinystories_instruction_tuning_plan_bn.md`: Complete Bengali reference mirror.
