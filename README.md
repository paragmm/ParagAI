# ParagAI 🚀

A modern, production-grade AI language model designed and built from scratch in Python from first principles.

---

## Architecture Overview

* **Dual-Tier Design:**
  * **Tier 1:** Pure Python & NumPy scalar autograd engine (`src/paragai/engine/`).
  * **Tier 2:** Modern Decoder-Only Transformer (`src/paragai/model/`) featuring:
    * Rotary Position Embeddings (RoPE)
    * Root Mean Square Normalization (RMSNorm)
    * Causal Multi-Head Self-Attention
    * SwiGLU Gated Feed-Forward Layers
    * Tied Input/Output Embeddings
* **Ollama Compatible:** Ready for GGUF conversion (`gguf-py`) and local execution via `ollama run paragai`.

---

## Quickstart

### 1. Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 2. Install Dependencies
```powershell
pip install -e .
```

### 3. Run Tests
```powershell
pytest tests/ -v
```

### 4. Train on Free Google Colab GPU (Optional)
Run on a free NVIDIA T4 GPU in 1 click using the included notebook:
* Open Google Colab and load: `notebooks/train_colab.ipynb`
* Or clone `https://github.com/paragmm/ParagAI.git` directly in Colab.

---

## Directory Structure
```text
ParagAI/
├── .agents/             # Agent guidelines & specialized skills
├── .plans/              # Architecture, training, and deployment plans
├── configs/             # YAML configurations for models and training
├── data/
│   ├── raw/             # Raw text datasets
│   └── processed/       # Tokenized memory-mapped datasets
├── checkpoints/         # Saved model weights (.pt, .safetensors, .gguf)
├── deploy/              # Modelfile and deployment configs for Ollama
├── src/
│   └── paragai/
│       ├── config.py    # Pydantic schemas
│       ├── engine/      # Tier 1 autograd engine
│       ├── tokenizer/   # BPE and Character tokenizers
│       ├── model/       # Transformer layers (Attention, RoPE, RMSNorm)
│       ├── training/    # Trainer, optimizer, and dataloaders
│       ├── inference/   # Generation & sampling (top-k, top-p, temperature)
│       └── utils/       # Seed reproducibility & logging
├── tests/               # Pytest unit tests
└── scripts/             # CLI entrypoints (train, chat, export)
```

