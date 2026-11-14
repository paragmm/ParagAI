# ParagAI 🧠

> **ParagAI — A first-principles LLM built from scratch. Bilingual (English + Indic). Runs on Ollama & vLLM.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)](tests/)

ParagAI is a fully transparent, educational, and production-ready large language model (LLM) implementation. Every component — from the autograd engine to the BPE tokenizer, to the LLaMA-style Transformer, to Ollama/vLLM deployment — is written from scratch. No black-box `transformers` library.

---

## ✨ Feature Highlights

| Feature | Status | Details |
|---|---|---|
| **Custom Autograd Engine** | ✅ | Pure Python + NumPy scalar computation graph |
| **LLaMA-Style Architecture** | ✅ | RoPE, RMSNorm, SwiGLU, Causal Attention |
| **Indic BPE Tokenizer** | ✅ | English + Bengali + Hindi + Tamil + Telugu |
| **`torch.compile`** | ✅ | 20-40% speedup on Linux/CUDA |
| **bfloat16 Training** | ✅ | Lower VRAM, same quality |
| **Distributed Training (DDP)** | ✅ | Multi-GPU via `torchrun` |
| **Ollama Deployment** | ✅ | `ollama run paragai` in one command |
| **vLLM Serving** | ✅ | OpenAI-compatible REST API |
| **Evaluation Suite** | ✅ | Perplexity, multilingual, story, tool adherence |

---

## 🏗️ Architecture Overview

```
ParagAI Stack
─────────────────────────────────────────────────────────
 Layer 0 ─ Autograd Engine        src/paragai/engine/
            (Pure Python, scalar Value graph, backprop)

 Layer 1 ─ Tokenizer              src/paragai/tokenizer/
            ┌──────────────────┐  ┌─────────────────────┐
            │  BPETokenizer    │  │  IndicBPETokenizer  │
            │  (English-only)  │  │  (Multilingual BPE) │
            └──────────────────┘  └─────────────────────┘

 Layer 2 ─ Model                  src/paragai/model/
            Embedding → [×N Transformer Blocks] → LM Head
            Each block:
              RMSNorm → CausalSelfAttention (RoPE) → RMSNorm
                      → SwiGLU FFN

 Layer 3 ─ Training               src/paragai/training/
            Trainer (AdamW, cosine LR, grad clip, AMP)
            TrainingTelemetry (tokens/sec, VRAM, ETA)

 Layer 4 ─ Export & Serve         scripts/
            export_hf.py  →  SafeTensors + HF config
            export_gguf.py →  GGUF  →  ollama create
            serve_vllm.sh  →  vLLM REST API (:8000)
─────────────────────────────────────────────────────────
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/paragmm/ParagAI.git
cd ParagAI

# 2. Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install (editable, with dev tools)
pip install -e ".[dev]"

# 4. Verify system
python scripts/doctor.py

# 5. Train a 5-minute demo
python scripts/train.py \
  --data data/raw/chat/conversations_sft.txt \
  --max_iters 500

# 6. Chat
python scripts/chat.py

# 7. Deploy to Ollama
python scripts/export_hf.py
python scripts/export_gguf.py
ollama create paragai -f deploy/Modelfile
ollama run paragai
```

---

## 📚 Documentation & Tutorials

| Guide (English) | গাইড (বাংলা) | Description |
|---|---|---|
| [01 — Quick Start](document/01_quickstart.md) | [০১ — দ্রুত শুরু গাইড](document/01_quickstart_bn.md) | Full setup, training, chat, and Ollama export |
| [02 — Training Guide](document/02_training_guide.md) | [০২ — ট্রেনিং গাইড](document/02_training_guide_bn.md) | 3-stage training, torch.compile, bfloat16, DDP multi-GPU |
| [03 — Tokenizer Guide](document/03_tokenizer_guide.md) | [০৩ — টোকেনাইজার গাইড](document/03_tokenizer_guide_bn.md) | IndicBPETokenizer, 256-byte fallback, Unicode subwords |
| [04 — Deployment Guide](document/04_deployment_guide.md) | [০৪ — ডিপ্লয়মেন্ট গাইড](document/04_deployment_guide_bn.md) | Ollama (GGUF) edge & vLLM (SafeTensors) cloud serving |
| [05 — Evaluation Guide](document/05_evaluation_guide.md) | [০৫ — ইভালুয়েশন গাইড](document/05_evaluation_guide_bn.md) | Benchmark suite: perplexity, multilingual, tool adherence |
| [06 — Continuous Training](document/06_continuous_training_and_data_preservation.md) | [০৬ — প্রতিদিনের ট্রেনিং ও জ্ঞান সংরক্ষণ](document/06_continuous_training_and_data_preservation_bn.md) | Incremental training without catastrophic forgetting |

---

## 🗂️ Project Structure

```
ParagAI/
├── configs/                  # YAML hyperparameter configs
│   ├── model_small.yaml      # Tiny model (n_embd=192, 6 layers)
│   ├── model_medium.yaml     # Medium model
│   └── train_config.yaml     # Training hyperparameters
├── data/
│   ├── raw/                  # Raw text data (put your .txt files here)
│   └── processed/            # Merged/tokenized splits
├── deploy/
│   └── Modelfile             # Ollama Modelfile (ChatML template)
├── document/                 # Tutorials and guides
├── notebooks/
│   └── train_colab.ipynb     # Google Colab notebook
├── scripts/
│   ├── doctor.py             # Pre-flight diagnostic checker
│   ├── train.py              # Main training script
│   ├── chat.py               # Interactive chat REPL
│   ├── export_hf.py          # Export to SafeTensors/HF format
│   ├── export_gguf.py        # Export to GGUF for Ollama
│   ├── eval.py               # Evaluation suite
│   └── train_multilingual_bpe.py  # Train IndicBPE tokenizer
├── src/paragai/
│   ├── config.py             # Pydantic config dataclasses
│   ├── engine/               # Pure Python autograd engine
│   ├── model/                # PyTorch Transformer (LLaMA-style)
│   ├── tokenizer/            # BPE + IndicBPE tokenizers
│   ├── training/             # Trainer, Dataset, Telemetry
│   └── utils/                # Seed, logger, helpers
└── tests/                    # pytest test suite
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output: all tests pass, covering autograd gradients, model shape invariants, causal masking, tokenizer roundtrip fidelity, and generation.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write code with type annotations, docstrings, and tests
4. Run `ruff check src/ scripts/` and `black src/ scripts/`
5. Submit a pull request

---

## 📖 Citing

```bibtex
@software{paragai2024,
  author  = {ParagAI Team},
  title   = {ParagAI: A First-Principles LLM Built From Scratch},
  year    = {2024},
  url     = {https://github.com/paragmm/ParagAI}
}
```

---

## 📄 License

MIT © ParagAI Team
