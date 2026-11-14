# ParagAI Tutorial 04: Edge (Ollama) & Cloud (vLLM) Deployment Guide

This guide covers universal serving of ParagAI models across edge environments (laptops/workstations with Ollama) and production servers (high-concurrency clouds with vLLM).

---

## 1. Edge Deployment with Ollama

Ollama runs models using llama.cpp with ultra-low memory overhead on CPUs, Apple Silicon, and consumer GPUs.

### Step 1: Export Checkpoint to SafeTensors
```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/best_model.pt \
  --output_dir checkpoints/export_hf \
  --verify
```
This generates:
- `checkpoints/export_hf/model.safetensors`
- `config.json` (LLaMA architecture specification)
- `generation_config.json`
- `tokenizer_config.json` (ChatML prompt formatting)
- `special_tokens_map.json`

### Step 2: Convert to GGUF with GPT-2 Byte Mapping
```bash
python scripts/export_gguf.py \
  --input_dir checkpoints/export_hf \
  --tokenizer checkpoints/tokenizer.json \
  --output checkpoints/paragai.gguf
```
> **Note on Token Byte Mapping**: ParagAI's GGUF exporter automatically maps whitespace and newline bytes using standard GPT-2 Unicode byte mappings (`\n` → `Ċ`, ` ` → `Ġ`). This guarantees clean, fluent English and Bengali text generation with zero `[UNK_BYTE_0x0a]` or `[UNK_BYTE_0x20]` artifacts in Ollama!

### Step 3: Register and Run in Ollama
Ensure your `deploy/Modelfile` points to the exported GGUF file:
```dockerfile
FROM ../checkpoints/paragai.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|endoftext|>"

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
```

Register into Ollama:
```bash
ollama create paragai -f deploy/Modelfile
```

Run interactive chat or query via CLI:
```bash
ollama run paragai:latest "Hello! Who are you?"
ollama run paragai:latest "একটি ছোট রূপকথার গল্প বলো।"
```

Query via REST API:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "paragai:latest",
  "prompt": "What is Python?",
  "stream": false
}'
```

---

## 2. Cloud Server Serving with vLLM

vLLM provides state-of-the-art serving throughput using PagedAttention and continuous batching.

### Step 1: Launch vLLM Server
On Linux / macOS:
```bash
bash scripts/serve_vllm.sh checkpoints/export_hf 8000
```

On Windows / WSL:
```bat
scripts\serve_vllm.bat checkpoints/export_hf 8000
```

### Step 2: Query the OpenAI-Compatible API
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "checkpoints/export_hf",
    "messages": [
      {"role": "system", "content": "You are ParagAI, a helpful AI assistant."},
      {"role": "user", "content": "Write a Python function to compute fibonacci."}
    ],
    "temperature": 0.3
  }'
```
