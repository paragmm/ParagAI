# ParagAI: Professional-Grade Multi-Task AI Roadmap

> **Document:** Enterprise-Grade Engineering & Training Roadmap  
> **Target:** Multi-Task AI System capable of **Chat**, **Language Translation (EN $\leftrightarrow$ BN)**, and **Autonomous Automation (Tool Calling)**  
> **Location:** `.plans/professional_grade_roadmap.md`  
> **Status:** Strategic Master Plan  

---

## 1. Vision & System Architecture

The objective of **ParagAI 2.0** is to transform from an experimental autoregressive decoder into a **production-grade multi-task AI model**. Rather than behaving like an uncontrollable raw text completer, the model is trained to recognize distinct operational modes:

```text
                               ┌─────────────────────────────────────────┐
                               │             USER PROMPT                 │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │      ParagAI Multi-Task Router          │
                               └────────────────────┬────────────────────┘
                                                    │
             ┌──────────────────────────────────────┼──────────────────────────────────────┐
             │                                      │                                      │
             ▼                                      ▼                                      ▼
┌─────────────────────────┐            ┌─────────────────────────┐            ┌─────────────────────────┐
│     MODE 1: CHATBOT     │            │    MODE 2: TRANSLATE    │            │   MODE 3: AUTOMATION    │
│  • Multi-turn Dialogue  │            │  • English ➔ Bengali    │            │  • Tool Use / Functions │
│  • Helpful Assistant    │            │  • Bengali ➔ English    │            │  • JSON Schema Actions  │
│  • Memory & Persona     │            │  • Technical Glossaries │            │  • Shell / API Trigger  │
└─────────────────────────┘            └─────────────────────────┘            └─────────────────────────┘
```

---

## 2. The 8-Phase Step-by-Step Implementation Roadmap

```text
[ Phase 1: Multi-Task Data Engine ]
        │ (Chat, Parallel Translation, and Tool Calling JSON datasets)
        ▼
[ Phase 2: Bilingual BPE Tokenizer (8K-16K Vocab) ]
        │ (English, Bengali Unicode, Code, and Special Control Tokens)
        ▼
[ Phase 3: High-Performance Architecture (KV-Cache + SDPA) ]
        │ (18M - 45M parameters, O(1) inference speed)
        ▼
[ Phase 4: Two-Stage Training Protocol ]
        │ (Stage A: General Pre-training ➔ Stage B: Multi-Task SFT)
        ▼
[ Phase 5: Task 1 — Chatbot Engine & Context Memory ]
        │ (Multi-turn conversations & ChatML formatting)
        ▼
[ Phase 6: Task 2 — Bilingual Translation (EN ⇄ BN) ]
        │ (High-precision parallel corpus alignment)
        ▼
[ Phase 7: Task 3 — Automation & Function Calling ]
        │ (Structured JSON action output & Python/Bash execution)
        ▼
[ Phase 8: Production Serving & Ollama GGUF Export ]
        │ (FastAPI OpenAI-compatible REST server, Web UI & Desktop deployment)
```

---

## 3. Phase Details & Technical Specifications

### Phase 1: Unified Multi-Task Prompt Schema

All tasks share a standardized **ChatML + Task Routing** template. This prevents the model from hallucinating or confusing instructions with conversations:

```text
<|im_start|>system
You are ParagAI, a professional AI system.
Task Capabilities: [chat, translation, automation]<|im_end|>

<|im_start|>user
[TASK: translate | source: en | target: bn]
Artificial Intelligence is transforming modern industries.<|im_end|>

<|im_start|>assistant
কৃত্রিম বুদ্ধিমত্তা আধুনিক শিল্প খাতকে রূপান্তরিত করছে।<|im_end|>
```

#### Automation / Tool-Calling Schema:
```text
<|im_start|>user
[TASK: automation]
Find all python files modified in the last 24 hours.<|im_end|>

<|im_start|>assistant
<|tool_call|>
{"tool": "run_shell", "parameters": {"command": "git status --short"}}
<|im_end|>
```

---

### Phase 2: Bilingual BPE Tokenizer (8K Vocab)

* **Current Limitation:** The character tokenizer breaks Bengali words into disjointed graphemes and wastes sequence capacity.
* **Upgrade:** Train our custom `BPETokenizer` on a balanced multilingual corpus:
  - 40% High-quality English text
  - 40% Bengali literary & technical prose
  - 20% Code, JSON syntax, and special control tokens
* **Reserved Control Tokens:**
  - `<|im_start|>`, `<|im_end|>` (Conversation turns)
  - `<|tool_call|>`, `<|tool_return|>` (Automation execution)
  - `[TASK: chat]`, `[TASK: translate]`, `[TASK: automation]`

---

### Phase 3: Core Architectural Enhancements

To make the model responsive like commercial AI models, we upgrade the attention block:

1. **Key-Value (KV) Caching:**
   - Instead of recalculating past keys and values on every word ($O(T^2)$), store past activations in GPU RAM.
   - **Result:** Text generation speed increases by **15x to 30x**.
2. **FlashAttention / PyTorch SDPA:**
   - Replace manual dot-product attention with `torch.nn.functional.scaled_dot_product_attention`.
   - **Result:** Memory usage drops by 50%; training speed triples.
3. **Scaled Parameter Profile (`ParagAI-Pro`):**
   - Embedding Dimension: $d_{\text{model}} = 384$
   - Layers: $10$ Decoder Blocks
   - Heads: $8$ Attention Heads
   - Context Window: $T = 512$ tokens
   - Parameters: **~28 Million** (Lightweight enough to run locally on CPU, yet smart enough to translate and follow instructions).

---

### Phase 4: Two-Stage Training Protocol

```
[ Raw Multilingual Text (15 MB - 50 MB) ] ──► Stage 1: Pre-training (Grammar & Word Associations)
                                                                 │
                                                                 ▼
[ 15,000 Multi-Task Instruction Pairs ]   ──► Stage 2: Supervised Fine-Tuning (SFT)
                                                                 │
                                                                 ▼
                                                  [ Production Model Checkpoint ]
```

* **Stage 1 (Pre-training):** Teaches the model vocabulary, language fluency, and common sense.
* **Stage 2 (Multi-Task SFT):** Teaches the model obedience, structured translation, and JSON tool generation.

---

### Phase 5: Production Chatbot Engine

* **Multi-Turn Context Management:** A sliding window memory class (`ConversationManager`) maintains conversation history up to the context limit.
* **System Prompt Personas:** Pre-configured personalities (e.g. "Coding Assistant", "Helpful Tutor", "Concise Bot").
* **Stop-Token Enforcement:** Automatically halts generation when `<|im_end|>` is emitted, preventing infinite repetitive loops.

---

### Phase 6: Bilingual Translation Engine (English $\leftrightarrow$ Bengali)

* **Dataset Source:** Pre-aligned parallel sentence pairs from **OPUS-100**, **IndicTrans2**, and curated bilingual dictionaries (~10,000 pairs).
* **Format:**
  ```text
  [TASK: translate | source: bn | target: en]
  আমি প্রতিদিন নতুন কিছু শেখার চেষ্টা করি।
  --> I try to learn something new every day.
  ```
* **Evaluation Metric:** Automated BLEU score validation to measure translation accuracy.

---

### Phase 7: Automation & Function Calling (Agentic AI)

To perform real-world automation, ParagAI outputs structured JSON actions that a Python wrapper executes:

```
[ User: "Show me git status" ]
               │
               ▼
   [ ParagAI Model Generates ] ──► `{"action": "shell", "command": "git status"}`
               │
               ▼
   [ Python Runner Executes ]  ──► Runs command in local workspace
               │
               ▼
   [ ParagAI Summarizes ]      ──► "You have 2 modified files: README.md and train.py."
```

#### Supported Local Tools:
1. `run_shell`: Run terminal commands safely with sandboxing.
2. `read_file` / `write_file`: Read or modify local workspace files.
3. `search_web`: Fetch external information.
4. `calculator`: Perform exact mathematical computations.

---

### Phase 8: Enterprise Serving & Deployment

1. **FastAPI OpenAI-Compatible Server (`scripts/serve.py`):**
   - Endpoints: `POST /v1/chat/completions` and `GET /v1/models`.
   - Allows drop-in replacement into any commercial AI frontend (LibreChat, Open WebUI, Continue.dev for VS Code).
2. **GGUF Quantization (`scripts/export_gguf.py`):**
   - Quantize the 28M parameter model to **`Q4_K_M`** (~18 MB file).
   - Runs with zero latency on any lightweight laptop, Raspberry Pi, or phone via Ollama (`ollama run paragai`).
3. **Interactive Web UI:**
   - A modern Streamlit / Gradio web dashboard for testing chat, translation, and tool automation with toggles.

---

## 4. Immediate Actionable First Step

To begin this transformation right now:
1. **Milestone 1:** Build the **Multi-Task Dataset Collector** (`scripts/collect_multitask_data.py`) to download:
   - 5,000 clean Chat dialogues
   - 5,000 English-Bengali translation sentence pairs
   - 1,000 Tool/Automation JSON examples
2. **Milestone 2:** Implement the **KV-Cache** in `src/paragai/model/attention.py`.
3. **Milestone 3:** Train the 28M parameter model on Google Colab GPU.

