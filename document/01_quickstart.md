# ParagAI Tutorial 01: Quickstart & Setup Guide

Welcome to **ParagAI**! This guide walks you through setting up your environment, running system diagnostics, performing your first training run, testing interactive chat, and deploying to Ollama and vLLM.

---

## 1. Environment Setup

### Prerequisites
* Python 3.10+ (Recommended: Python 3.11)
* Git
* Recommended for local serving: [Ollama](https://ollama.com/)

### Clone and Virtual Environment
```bash
# Clone the repository
git clone https://github.com/paragmm/ParagAI.git
cd ParagAI

# Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Linux / macOS:
source .venv/bin/activate

# Install ParagAI in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## 2. Run Pre-Flight System Diagnostics

ParagAI includes an automated diagnostic suite `scripts/doctor.py` that verifies your PyTorch version, CUDA GPU availability, VRAM, `torch.compile` status, checkpoint health, tokenizer roundtrip fidelity, and Ollama service:

```bash
python scripts/doctor.py
```

Expected output:
```
                       ParagAI Doctor — Diagnostic Summary                     
┌────────┬──────────────────────────────┬──────────┬───────────────────────────┐
│ Status │ Check                        │ Critical │ Detail                    │
├────────┼──────────────────────────────┼──────────┼───────────────────────────┤
│   ✓    │ Python version ≥ 3.10        │   yes    │ Python 3.11.9             │
│   ✓    │ PyTorch + CUDA               │    no    │ torch 2.15.0 | CUDA ✓     │
│   ✓    │ torch.compile                │    no    │ Available                 │
│   ✓    │ Checkpoint + config          │    no    │ 58 tensors | 2.7M params  │
│   ✓    │ Tokenizer roundtrip          │    no    │ Roundtrip verified        │
│   ✓    │ Ollama service               │    no    │ Reachable                 │
└────────┴──────────────────────────────┴──────────┴───────────────────────────┘
✓ All critical checks passed.
```

---

## 3. Your First Training Run

ParagAI uses modular configuration files under `configs/`:
- `configs/model_small.yaml`: Compact 2.7M parameter transformer (6 layers, 192 embedding dim, 6 attention heads).
- `configs/train_fast.yaml`: Fast training with gradient accumulation.

To start training with the live telemetry dashboard:
```bash
python scripts/train.py \
  --data data/raw/chat/conversations_sft.txt \
  --config configs/train_config.yaml \
  --max_iters 500
```

The live dashboard displays in your terminal:
```
╔══════════════════════════════════════════════════╗
║            ParagAI  Training  Dashboard          ║
╠══════════════════════════════════════════════════╣
│  Step               500                          │
│  Elapsed            00:01:23                     │
│  ETA                00:00:00                     │
│  Tokens/sec         14,250                       │
│  VRAM (MB)          1,840.5                      │
╠══════════════════════════════════════════════════╣
│  Learning Rate      1.5000e-05                   │
│  Grad Norm          0.4120                       │
│  Train Loss         0.0245                       │
╠══════════════════════════════════════════════════╣
│  Val Loss           0.0287                       │
│  Perplexity         1.03                         │
╚══════════════════════════════════════════════════╝
```

---

## 4. Chat Interactively with the Model

Test the model in your terminal using the ChatML-compliant REPL:
```bash
python scripts/chat.py --checkpoint checkpoints/best_model.pt --tokenizer checkpoints/tokenizer.json
```

Example interaction:
```
ParagAI > Hello!
Assistant > Hello! I am ParagAI, your personal assistant. How can I assist you today?

ParagAI > হ্যালো, আপনি কেমন আছেন?
Assistant > আমি ভালো আছি! আপনি কেমন আছেন? আপনাকে কিভাবে সাহায্য করতে পারি?
```

---

## 5. Deploy to Edge (Ollama)

Convert your trained checkpoint to HuggingFace SafeTensors, compile it to GGUF, and register it in Ollama:

```bash
# 1. Export to HuggingFace format
python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output_dir checkpoints/export_hf --verify

# 2. Export to GGUF with GPT-2 byte mapping
python scripts/export_gguf.py --input_dir checkpoints/export_hf --tokenizer checkpoints/tokenizer.json --output checkpoints/paragai.gguf

# 3. Register model into Ollama
ollama create paragai -f deploy/Modelfile

# 4. Run directly
ollama run paragai:latest "Write a story about a brave astronaut."
```

---

## 6. Deploy to Cloud / Server (vLLM)

For high-throughput enterprise serving with continuous batching and PagedAttention:

```bash
# On Linux / macOS:
bash scripts/serve_vllm.sh checkpoints/export_hf 8000

# On Windows:
scripts\serve_vllm.bat checkpoints/export_hf 8000
```

Access the standard OpenAI-compatible API at:
`http://localhost:8000/v1/chat/completions`
