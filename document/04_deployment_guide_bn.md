# ParagAI টিউটোরিয়াল ০৪: লোকাল (Ollama) ও ক্লাউড (vLLM) ডিপ্লয়মেন্ট গাইড

এই গাইডে শেখানো হয়েছে কীভাবে ParagAI মডেলকে আপনার নিজের ল্যাপটপে Ollama দিয়ে এবং ক্লাউড সার্ভারে vLLM দিয়ে চালু করবেন।

---

## ১. লোকাল কম্পিউটারে Ollama-র মাধ্যমে চালানো

Ollama অত্যন্ত কম র‍্যাম ব্যবহার করে আপনার নিজের পিসিতে মডেল চালাতে সাহায্য করে।

### ধাপ ১: SafeTensors ফরম্যাটে এক্সপোর্ট করা
```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/best_model.pt \
  --output_dir checkpoints/export_hf \
  --verify
```

### ধাপ ২: GGUF ফাইল তৈরি করা
```bash
python scripts/export_gguf.py \
  --input_dir checkpoints/export_hf \
  --tokenizer checkpoints/tokenizer.json \
  --output checkpoints/paragai.gguf
```
> **জরুরি বিষয়:** ParagAI এর GGUF এক্সপোর্টার স্বয়ংক্রিয়ভাবে স্পেস এবং নিউলাইন ক্যারেক্টারকে GPT-2 ইউনিকোড ম্যাপ করে। ফলে ওলামাতে কোনো `[UNK_BYTE]` এর ত্রুটি আসে না।

### ধাপ ৩: ওলামাতে মডেল রেজিস্টার করা
```bash
ollama create paragai -f deploy/Modelfile
```

### ধাপ ৪: ওলামা দিয়ে মডেল টেস্ট করা
```bash
ollama run paragai:latest "Hello! Who are you?"
ollama run paragai:latest "একটি ছোট রূপকথার গল্প বলো।"
```

---

## ২. ক্লাউড ও এন্টারপ্রাইজ সার্ভারে vLLM চালানো

সার্ভারে হাজার হাজার ইউজারের জন্য এক সাথে দ্রুত রেসপন্স দিতে vLLM ব্যবহার করা হয়:

```bash
# লিনাক্সে:
bash scripts/serve_vllm.sh checkpoints/export_hf 8000

# উইন্ডোজে:
scripts\serve_vllm.bat checkpoints/export_hf 8000
```

সার্ভার চালু হলে সরাসরি `http://localhost:8000/v1/chat/completions` এ OpenAI API ফরম্যাটে কল করা যায়।
