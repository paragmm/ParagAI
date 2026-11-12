# ParagAI: Ollama Deployment & GGUF Conversion Plan

> **Document:** End-to-End Guide to Export, Convert, and Run ParagAI on Ollama  
> **Location:** `.plans/ollama_export_plan.md`  
> **Target:** Native C++ execution, quantization (Q4/Q8), and OpenAI-compatible API serving  
> **Status:** Ready for Execution  

---

## 1. Executive Summary & Objective

Ollama is a high-performance local AI runtime built on top of **llama.cpp**. It cannot directly execute raw Python `.py` files or proprietary PyTorch `.pt` checkpoint files. Instead, it expects models in the standardized **GGUF (GPT-Generated Unified Format)** container.

The goal of this plan is to establish a seamless, automated pipeline:
$$\text{ParagAI (PyTorch Checkpoint)} \xrightarrow[\text{Script}]{\text{Export}} \text{HuggingFace / SafeTensors} \xrightarrow[\text{gguf-py}]{\text{Convert}} \text{ParagAI.gguf} \xrightarrow[\text{Modelfile}]{\text{Ollama}} \text{ollama run paragai}$$

---

## 2. Technical Prerequisites: Architectural Compatibility

For Ollama / `llama.cpp` to recognize and execute our custom model without requiring custom C++ kernel patches, **ParagAI** is architected to follow standard LLaMA-compatible conventions:

| Component | ParagAI Choice | Ollama / LLaMA Equivalent |
|---|---|---|
| **Architecture Tag** | Decoder-Only Transformer | `LlamaForCausalLM` |
| **Attention** | Scaled Dot-Product Multi-Head | Self-Attention with Causal Mask |
| **Positional Encoding** | Rotary Positional Embedding (RoPE) | Standard RoPE (`freq_base = 10000.0`) |
| **Normalization** | Root Mean Square Normalization | `RMSNorm` (without bias) |
| **Feed-Forward** | Gated Linear Unit (SwiGLU) | SwiGLU (`gate_proj`, `up_proj`, `down_proj`) |
| **Tied Embeddings** | Language Model Head tied to Embeddings | `tie_word_embeddings: true` |

---

## 3. The 6-Step Implementation Roadmap

```text
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│ Step 1: Weight & Config│ ───► │ Step 2: Tokenizer      │ ───► │ Step 3: GGUF           │
│ Standardization (HF)   │      │ Serialization          │      │ Conversion (gguf-py)   │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
            │                                                                │
            ▼                                                                ▼
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│ Step 6: REST API & UI  │ ◄─── │ Step 5: Ollama Create  │ ◄─── │ Step 4: Modelfile      │
│ Integration Testing    │      │ & CLI Execution        │      │ Construction           │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

---

### Step 1: Weight Mapping & Config Export (`scripts/export_to_hf.py`)

Ollama's converters look for standard tensor names. We create a script that converts our internal PyTorch `state_dict` keys into standard naming conventions:

#### Tensor Mapping Table:
| ParagAI Internal Key | Standard Target Key |
|---|---|
| `token_embedding.weight` | `model.embed_tokens.weight` |
| `blocks.{i}.attn.q_proj.weight` | `model.layers.{i}.self_attn.q_proj.weight` |
| `blocks.{i}.attn.k_proj.weight` | `model.layers.{i}.self_attn.k_proj.weight` |
| `blocks.{i}.attn.v_proj.weight` | `model.layers.{i}.self_attn.v_proj.weight` |
| `blocks.{i}.attn.out_proj.weight`| `model.layers.{i}.self_attn.o_proj.weight` |
| `blocks.{i}.ffn.w_gate.weight`  | `model.layers.{i}.mlp.gate_proj.weight` |
| `blocks.{i}.ffn.w_up.weight`    | `model.layers.{i}.mlp.up_proj.weight` |
| `blocks.{i}.ffn.w_down.weight`  | `model.layers.{i}.mlp.down_proj.weight` |
| `blocks.{i}.norm1.weight`       | `model.layers.{i}.input_layernorm.weight` |
| `blocks.{i}.norm2.weight`       | `model.layers.{i}.post_attention_layernorm.weight` |
| `final_norm.weight`             | `model.norm.weight` |
| `lm_head.weight`                | `lm_head.weight` |

The script outputs:
* `checkpoints/export_hf/model.safetensors` (zero-copy safe tensor format)
* `checkpoints/export_hf/config.json` (architecture metadata)

---

### Step 2: Tokenizer Serialization

GGUF files store the vocabulary and special tokens directly inside the model file so that the C++ runtime can tokenize without external Python dependencies.

* Export our vocabulary to standard `tokenizer.json` and `tokenizer_config.json`.
* Register mandatory special tokens:
  - `<unk>`: Unknown token (ID: 0)
  - `<s>` or `<bos>`: Beginning of sequence (ID: 1)
  - `</s>` or `<eos>`: End of sequence (ID: 2)

---

### Step 3: Direct GGUF Conversion (`scripts/export_to_gguf.py`)

Rather than relying on compiling large third-party C++ repositories, we use the official Python `gguf` package (`pip install gguf`):

```python
# Conceptual conversion using gguf-py
import gguf
from safetensors.torch import load_file

def convert_to_gguf(weights_path, config, output_path):
    writer = gguf.GGUFWriter(output_path, arch="llama")
    
    # 1. Write Model Metadata
    writer.add_name("ParagAI")
    writer.add_context_length(config["max_seq_len"])
    writer.add_embedding_length(config["n_embd"])
    writer.add_block_count(config["n_layer"])
    writer.add_feed_forward_length(config["n_embd"] * 4)
    writer.add_head_count(config["n_head"])
    writer.add_layer_norm_rms_eps(1e-5)
    
    # 2. Write Tokenizer Metadata
    writer.add_tokenizer_model("llama")
    writer.add_token_list(vocab_tokens)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    
    # 3. Write Tensors
    tensors = load_file(weights_path)
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor.numpy())
        
    writer.write_header_to_file()
    writer.write_data_to_file()
    writer.close()
```

Output: `checkpoints/paragai.gguf` (FP16 or FP32 precision).

---

### Step 4: Quantization (Optional but Recommended)

Once the GGUF file is produced, we can quantize the weights to drastically reduce memory and speed up CPU inference:

* **`Q8_0` (8-bit quantization):** 50% memory reduction with almost zero degradation in perplexity.
* **`Q4_K_M` (4-bit quantization):** 75% memory reduction, enabling 100+ tokens/sec on lightweight CPUs.

---

### Step 5: Ollama `Modelfile` Construction

Create `deploy/Modelfile` in your workspace:

```dockerfile
# 1. Base Model GGUF File
FROM ../checkpoints/paragai.gguf

# 2. Runtime Parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "</s>"
PARAMETER stop "<|endoftext|>"

# 3. System Prompt & Behavior
SYSTEM """
You are ParagAI, a compact, professional AI model designed and trained from scratch in Python.
Respond thoughtfully, concisely, and accurately.
"""

# 4. Chat Template
TEMPLATE """
{{ if .System }}<|system|>
{{ .System }}<|endoftext|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|endoftext|>
<|assistant|>
{{ end }}{{ .Response }}
"""
```

---

### Step 6: Registration, CLI Testing & REST API

#### 1. Register with Ollama
```bash
ollama create paragai -f ./deploy/Modelfile
```

#### 2. Verify Installation
```bash
ollama list
```
*(You will see `paragai:latest` with its size, e.g., ~50MB - 300MB)*

#### 3. Run Interactive CLI Chat
```bash
ollama run paragai
```

#### 4. Test via HTTP REST API (Standard Ollama & OpenAI Compatible)
```bash
# Using standard cURL
curl http://localhost:11434/api/generate -d '{
  "model": "paragai",
  "prompt": "Once upon a time",
  "stream": false
}'
```

---

## 4. Troubleshooting & Verification Matrix

| Issue | Root Cause | Solution |
|---|---|---|
| `Unknown architecture` | GGUF architecture tag is not recognized by llama.cpp. | Ensure architecture tag is set to `"llama"` in `gguf` writer. |
| `Tokenizer out of bounds` | Tokenizer vocabulary size in `config.json` does not match `embed_tokens.weight` matrix shape. | Ensure `vocab_size == embed_tokens.shape[0]`. |
| `Illegal instruction / crash` | Missing CPU AVX2 instructions or corrupted GGUF header. | Re-export with `outtype f16` or run on standard FP32 before quantizing. |
| `Ollama returns endless loop` | Stop tokens (`<eos>`) are not configured in `Modelfile`. | Add `PARAMETER stop "</s>"` and `PARAMETER repeat_penalty 1.1`. |

---

## 5. Execution Scripts Summary

By adding these 3 dedicated scripts to the project, the entire Ollama export becomes a 1-line command:

1. `scripts/export_hf.py`: Converts PyTorch checkpoint to HuggingFace format.
2. `scripts/export_gguf.py`: Bundles HuggingFace format and tokenizer into `.gguf`.
3. `scripts/setup_ollama.bat` (or `.sh`): Runs `ollama create paragai -f deploy/Modelfile`.

