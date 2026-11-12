# ParagAI: Complete Guide to Generating and Running Models in Ollama

> **Author:** ParagAI Engineering Team  
> **Location:** `document/ollama_model_generation_guide.md`  
> **Target Audience:** Developers deploying ParagAI to Ollama (CLI & Web UI)  

---

## 1. Overview: The Ollama Deployment Pipeline

Ollama runs models using the **GGUF** (GPT-Generated Unified Format) container format powered by `llama.cpp`. Since `ParagAI` is a PyTorch-native Transformer built from first principles, we convert it through a standardized 3-stage export pipeline:

```
┌─────────────────────────┐
│ ParagAI PyTorch Model   │  (checkpoints/best_model.pt)
│ State Dict & Config     │
└───────────┬─────────────┘
            │
            ▼  Step 1: Convert to HuggingFace / LLaMA format
┌─────────────────────────┐
│ SafeTensors + Config    │  (checkpoints/export_hf/model.safetensors)
│ Standard LLaMA Weights  │  (checkpoints/export_hf/config.json)
└───────────┬─────────────┘
            │
            ▼  Step 2: Package into GGUF Container
┌─────────────────────────┐
│ GGUF Binary Model File  │  (checkpoints/paragai.gguf)
│ Quantizable & Portable  │
└───────────┬─────────────┘
            │
            ▼  Step 3: Register with Ollama Modelfile
┌─────────────────────────┐
│ Local Ollama AI Engine  │  (ollama run paragai)
│ Terminal & REST API     │  (OpenAI-compatible localhost:11434)
└─────────────────────────┘
```

---

## 2. Prerequisites

1. **Python Environment Activated:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. **Required Libraries Installed:**
   Ensure `safetensors` and `gguf` are installed in your `.venv`:
   ```powershell
   pip install safetensors gguf
   ```
3. **Ollama Installed on Windows:**
   - Ollama is located at: `C:\Users\hedpa\AppData\Local\Programs\Ollama\ollama.exe`
   - Make sure the Ollama desktop app is running in your Windows system tray, or start it via:
     ```powershell
     ollama serve
     ```

---

## 3. Step-by-Step Execution Guide

### Step 1: Export PyTorch Checkpoint to SafeTensors
Converts your internal PyTorch `.pt` file to the standard HuggingFace / LLaMA tensor naming convention:

```powershell
python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output_dir checkpoints/export_hf
```

* **What this does:**
  - Extracts weights and maps them to standard LLaMA keys (`model.layers.X.self_attn`, `mlp.gate_proj`, etc.).
  - Clones shared embedding weights to prevent memory-tying issues.
  - Outputs `checkpoints/export_hf/model.safetensors` and `checkpoints/export_hf/config.json`.

---

### Step 2: Convert SafeTensors to GGUF Binary
Packages the weights and vocabulary into a single `.gguf` file that Ollama can load into CPU/GPU memory:

```powershell
python scripts/export_gguf.py --input_dir checkpoints/export_hf --tokenizer checkpoints/tokenizer.json --output checkpoints/paragai.gguf
```

* **What this does:**
  - Writes model architecture metadata (`llama`, layer count, head count, embedding dimensions).
  - Encodes the vocabulary and special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).
  - Serializes all tensor weights in 16-bit float (`f16`).
  - Outputs `checkpoints/paragai.gguf` (approx. 10.8 MB for small prototype).

---

### Step 3: Configure the Ollama `Modelfile`
The `deploy/Modelfile` defines the model system prompt, temperature, stop tokens, and chat template.

Verify or edit [`deploy/Modelfile`](file:///c:/Users/hedpa/Documents/AI/ParagAI/deploy/Modelfile):
```dockerfile
# Point to your generated GGUF file
FROM ../checkpoints/paragai.gguf

# Generation Hyperparameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Stop Tokens
PARAMETER stop "</s>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"

# System Persona
SYSTEM """
You are ParagAI, a compact, intelligent AI assistant trained from first principles in Python.
Respond thoughtfully, concisely, and accurately.
"""

# Chat Template (ChatML)
TEMPLATE """
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}{{ .Response }}<|im_end|>
"""
```

---

### Step 4: Register the Model into Ollama
Run this command from your project root directory:

```powershell
ollama create paragai -f deploy/Modelfile
```

* **Expected Output:**
  ```text
  transferring model data
  creating model layer
  using already created layer
  writing manifest
  success
  ```

---

### Step 5: Verify & Run the Model in Ollama

#### 1. Check Registered Models:
```powershell
ollama list
```
You should see `paragai:latest` in the list.

#### 2. Start an Interactive Chat in the Terminal:
```powershell
ollama run paragai
```
Type any prompt and test your model! To exit the chat, type `/bye` or press `Ctrl+D`.

#### 3. Single-Line Generation Command:
```powershell
ollama run paragai "ParagAI is designed to"
```

---

## 4. Using the Model via REST API (Python & Web Apps)

Once registered in Ollama, your ParagAI model is instantly available via local HTTP APIs:

### Native Ollama REST API (Port 11434):
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "paragai",
  "prompt": "Hello ParagAI, what can you do?",
  "stream": false
}'
```

### OpenAI-Compatible API (for Open-WebUI, Chatbots, etc.):
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" # required by the library, but not checked
)

response = client.chat.completions.create(
    model="paragai",
    messages=[
        {"role": "system", "content": "You are ParagAI."},
        {"role": "user", "content": "Explain machine learning in simple terms."}
    ]
)

print(response.choices[0].message.content)
```

---

## 5. Troubleshooting & FAQs

### Q1: `Warning: could not connect to a running Ollama instance`
* **Fix:** Ollama background service is not running.
  - Launch Ollama from Windows Start Menu, or
  - Run `ollama serve` in a separate terminal.

### Q2: `Error: 'GGUFWriter' object has no attribute 'add_padding_token_id'`
* **Fix:** Make sure you use `scripts/export_gguf.py` which uses the updated `writer.add_pad_token_id()`.

### Q3: Why is the model output repetitive or short?
* **Fix:** The current checkpoint is a small base model (2.68M parameters) trained on raw text snippets. Once you train a BPE-based model with instruction datasets (as detailed in `04_model_intelligence_plan.md` and `05_professional_grade_roadmap.md`), the GGUF model in Ollama will produce rich, fluent, and intelligent answers.

---

## 6. One-Click Automation Script

To perform the entire export in a single command, you can run:

```powershell
# Export HF -> Export GGUF -> Register in Ollama
python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output_dir checkpoints/export_hf
python scripts/export_gguf.py --input_dir checkpoints/export_hf --tokenizer checkpoints/tokenizer.json --output checkpoints/paragai.gguf
ollama create paragai -f deploy/Modelfile
ollama run paragai "Hello!"
```

