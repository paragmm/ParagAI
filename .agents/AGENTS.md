# ParagAI Developer Agent Guidelines & Workspace Rules

Welcome to the **ParagAI** repository. These instructions govern all AI agent behaviors, code authoring standards, and operational guidelines across this workspace.

---

## 1. Project Philosophy

1. **First-Principles Understanding:** Implement all core algorithms (Autograd, Attention, RMSNorm, Tokenization, Generation) from scratch. Avoid using black-box libraries (`transformers`, `langchain`) for the model architecture itself.
2. **Production-Grade Clean Code:** All code must have explicit type annotations (`typing`), clear docstrings, reproducible random seeds, and unit tests.
3. **Dual-Tier Architecture:**
   - `src/paragai/engine/`: Pure Python & NumPy autograd engine (scalar graph).
   - `src/paragai/model/`: PyTorch-accelerated, LLaMA-compatible Transformer decoder.

---

## 2. Coding Standards & Conventions

* **Python Version:** 3.10+
* **Formatting & Linting:** `ruff` and `black` standards. Line length: 100 characters.
* **Typing:** Strict typing throughout `src/paragai/`. Use `torch.Tensor` or `NDArray` annotations where appropriate.
* **Configuration:** Never hardcode hyperparameters. Use Pydantic dataclasses (`src/paragai/config.py`) or YAML files under `configs/`.
* **Reproducibility:** Always set random seeds across Python, NumPy, and PyTorch using `src/paragai/utils/seed.py`:
  ```python
  from paragai.utils.seed import set_seed
  set_seed(42)
  ```

---

## 3. Testing Requirements

* Every new component or layer must be accompanied by a dedicated unit test in `tests/`.
* Tests must verify:
  1. **Shape Invariants:** (e.g. `(B, T, C)` $\to$ `(B, T, C)`).
  2. **Causal Masking:** No information leakage from future tokens.
  3. **Gradient Sanity:** Backpropagation produces non-zero gradients on all parameters.
  4. **Tokenizer Fidelity:** Roundtrip encoding/decoding is lossless (`decode(encode(text)) == text`).

Run tests using:
```bash
pytest tests/ -v
```

---

## 4. Specialized Skills Index

Refer to these specialized skills under `.agents/skills/`:
- **`model-architect`**: Designing, implementing, and debugging Transformer blocks and mathematical layers.
- **`tokenizer-builder`**: Training and evaluating BPE and character tokenizers.
- **`model-trainer`**: Hyperparameters, dataset batching, training loops, and loss monitoring.
- **`ollama-exporter`**: Converting PyTorch models to SafeTensors/GGUF and deploying to Ollama.

