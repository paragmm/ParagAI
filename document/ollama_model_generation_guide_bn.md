# ParagAI: Ollama মডেল তৈরি ও রান করার সম্পূর্ণ গাইডলাইন (Step-by-Step)

> **ডকুমেন্ট:** Ollama-তে ParagAI ডিপ্লয়মেন্ট এবং ব্যবহারের নির্দেশিকা  
> **লোকেশন:** `document/ollama_model_generation_guide_bn.md`  

---

## ১. ওভারভিউ: Ollama ডিপ্লয়মেন্ট পাইপলাইন

Ollama মূলত **GGUF** ফরম্যাট ব্যবহার করে মডেল লোড ও রান করে। যেহেতু ParagAI একটি PyTorch-নেটিভ ট্রান্সফরমার মডেল, তাই এটিকে Ollama-তে চালানোর জন্য ৩-ধাপের এক্সপোর্ট পাইপলাইন অনুসরণ করতে হয়:

```
┌─────────────────────────┐
│ ParagAI PyTorch Model   │  (checkpoints/best_model.pt)
│ State Dict & Config     │
└───────────┬─────────────┘
            │
            ▼  ধাপ ১: HuggingFace / LLaMA ফরম্যাটে রূপান্তর
┌─────────────────────────┐
│ SafeTensors + Config    │  (checkpoints/export_hf/model.safetensors)
│ Standard LLaMA Weights  │  (checkpoints/export_hf/config.json)
└───────────┬─────────────┘
            │
            ▼  ধাপ ২: GGUF বাইনারি কন্টেইনারে প্যাকেজিং
┌─────────────────────────┐
│ GGUF Binary Model File  │  (checkpoints/paragai.gguf)
│ Quantizable & Portable  │
└───────────┬─────────────┘
            │
            ▼  ধাপ ৩: Ollama Modelfile দিয়ে রেজিস্ট্রেশন
┌─────────────────────────┐
│ Local Ollama AI Engine  │  (ollama run paragai)
│ Terminal & REST API     │  (OpenAI-compatible localhost:11434)
└─────────────────────────┘
```

---

## ২. প্রয়োজনীয় পূর্বশর্ত (Prerequisites)

১. **ভার্চুয়াল এনভায়রনমেন্ট সচল করা:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
২. **প্রয়োজনীয় লাইব্রেরি ইনস্টল থাকা:**
   `safetensors` এবং `gguf` ইনস্টল থাকতে হবে:
   ```powershell
   pip install safetensors gguf
   ```
৩. **উইন্ডোজ মেশিনে Ollama চালু থাকা:**
   - আপনার কম্পিউটারে Ollama ইনস্টল করা আছে (`C:\Users\hedpa\AppData\Local\Programs\Ollama\ollama.exe`)।
   - নিশ্চিত করুন Ollama ব্যাকগ্রাউন্ডে চলছে। যদি চালু না থাকে, তবে টার্মিনালে কমান্ড দিন:
     ```powershell
     ollama serve
     ```
     অথবা উইন্ডোজের Start Menu থেকে Ollama অ্যাপটি ওপেন করুন।

---

## ৩. ধাপে ধাপে এক্সপোর্ট ও রেজিস্ট্রেশন প্রক্রিয়া

### ধাপ ১: PyTorch চেকপয়েন্টকে SafeTensors-এ রূপান্তর
এটি আপনার ইন্টারনাল PyTorch `.pt` ফাইলটিকে স্ট্যান্ডার্ড LLaMA ফরম্যাটে রূপান্তর করে:

```powershell
python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output_dir checkpoints/export_hf
```

* **কী ঘটবে:**
  - `checkpoints/export_hf/model.safetensors` এবং `config.json` তৈরি হবে।

---

### ধাপ ২: SafeTensors থেকে GGUF ফাইলে রূপান্তর
এটি মডেল ওয়েটস ও টোকেনাইজার ভোকাবুলারিকে Ollama-র উপযোগী একক `.gguf` ফাইলে রূপান্তর করে:

```powershell
python scripts/export_gguf.py --input_dir checkpoints/export_hf --tokenizer checkpoints/tokenizer.json --output checkpoints/paragai.gguf
```

* **কী ঘটবে:**
  - `checkpoints/paragai.gguf` ফাইল তৈরি হবে (যা আমরা ইতোমধ্যে সফলভাবে তৈরি করেছি, সাইজ প্রায় ১০.৮ MB)।

---

### ধাপ ৩: Ollama `Modelfile` কনফিগারেশন
[`deploy/Modelfile`](file:///c:/Users/hedpa/Documents/AI/ParagAI/deploy/Modelfile) ফাইলটি ওপেন করে নিশ্চিত করুন যে এটি সঠিক GGUF ফাইলকে নির্দেশ করছে:

```dockerfile
# তৈরি করা GGUF ফাইলের পাথ
FROM ../checkpoints/paragai.gguf

# জেনারেশন প্যারামিটার
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# স্টপ টোকেন
PARAMETER stop "</s>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"

# সিস্টেম প্রম্পট
SYSTEM """
You are ParagAI, a compact, intelligent AI assistant trained from first principles in Python.
Respond thoughtfully, concisely, and accurately.
"""

# চ্যাট টেমপ্লেট
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

### ধাপ ৪: Ollama-তে মডেল রেজিস্টার করা
প্রজেক্টের রুট ডিরেক্টরি থেকে টার্মিনালে এই কমান্ডটি চালান:

```powershell
ollama create paragai -f deploy/Modelfile
```

* **সফল আউটপুট কেমন দেখাবে:**
  ```text
  transferring model data
  creating model layer
  using already created layer
  writing manifest
  success
  ```

---

### ধাপ ৫: Ollama দিয়ে মডেল চালানো ও চ্যাট করা

#### ১. মডেল লিস্ট চেক করা:
```powershell
ollama list
```
তালিকায় `paragai:latest` দেখতে পাবেন।

#### ২. টার্মিনালে সরাসরি চ্যাট শুরু করা:
```powershell
ollama run paragai
```
চ্যাট শেষ করতে `/bye` লিখুন।

#### ৩. এক লাইনে প্রম্পট পাঠানো:
```powershell
ollama run paragai "ParagAI is designed to"
```

---

## ৪. REST API ও অন্যান্য অ্যাপে ব্যবহার (Python / Web UI)

Ollama ব্যাকগ্রাউন্ডে চললে আপনি সরাসরি লোকাল API-র মাধ্যমে ParagAI ব্যবহার করতে পারবেন:

### পাইথন স্ক্রিপ্টে OpenAI ফরম্যাটে কল করা:
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
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

## ৫. সব কাজ এক ক্লিকে করার শর্টকাট স্ক্রিপ্ট (One-Liner)

সব ধাপ একসাথে চালাতে নিচের কমান্ডগুলো একযোগে রান করতে পারেন:

```powershell
python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output_dir checkpoints/export_hf
python scripts/export_gguf.py --input_dir checkpoints/export_hf --tokenizer checkpoints/tokenizer.json --output checkpoints/paragai.gguf
ollama create paragai -f deploy/Modelfile
ollama run paragai
```
