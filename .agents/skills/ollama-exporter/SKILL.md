---
name: ollama-exporter
description: >-
  Use this skill when converting ParagAI PyTorch models to SafeTensors/GGUF format, configuring Ollama
  Modelfiles, registering models into Ollama, or testing Ollama CLI and REST API endpoints.
---

# ParagAI Ollama Exporter Skill

This skill provides procedures for converting trained `ParagAI` PyTorch checkpoints into standardized GGUF containers and deploying them directly into Ollama.

## Export & Deployment Sequence

Follow these 4 sequential steps to take a trained model into Ollama:

### Step 1: Export Weights to SafeTensors & LLaMA Config
Converts the internal model checkpoint into HuggingFace/LLaMA-standard tensor names:
```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/best_model.pt \
  --output_dir checkpoints/export_hf/
```
Verifies output contains:
- `model.safetensors`
- `config.json` (containing `"architectures": ["LlamaForCausalLM"]`)
- `tokenizer.json`

### Step 2: Convert SafeTensors to GGUF Container
Converts the standardized model and vocabulary into a `.gguf` file using `gguf-py`:
```bash
python scripts/export_gguf.py \
  --input_dir checkpoints/export_hf/ \
  --output checkpoints/paragai.gguf \
  --outtype f16
```

### Step 3: Configure Ollama Modelfile
Ensure `deploy/Modelfile` points to the generated GGUF:
```dockerfile
FROM ../checkpoints/paragai.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "</s>"
PARAMETER stop "<|endoftext|>"

SYSTEM """
You are ParagAI, a compact AI model designed and trained from scratch in Python.
"""
```

### Step 4: Register & Run with Ollama
```bash
# Register the model in Ollama
ollama create paragai -f deploy/Modelfile

# Verify registration
ollama list

# Test interactive chat
ollama run paragai "Explain what an autograd engine is in one sentence."
```

---

## REST API Verification
Verify that Ollama exposes both its native endpoint and the OpenAI-compatible endpoint:
```bash
# Native Ollama API
curl http://localhost:11434/api/generate -d "{\"model\": \"paragai\", \"prompt\": \"Hello!\", \"stream\": false}"

# OpenAI-Compatible API
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"paragai\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}"
```

