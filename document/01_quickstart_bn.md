# ParagAI টিউটোরিয়াল ০১: দ্রুত শুরু ও সেটআপ গাইড (Quickstart Guide)

**ParagAI** প্রজেক্টে আপনাকে স্বাগতম! এই গাইডে আমরা খুব সহজে পরিবেশ সেটআপ, সিস্টেম ডায়াগনস্টিক পরীক্ষা, প্রথম মডেল ট্রেনিং, ইন্টারঅ্যাক্টিভ চ্যাট এবং Ollama ও vLLM এ ডিপ্লয় করার সম্পূর্ণ ধাপ দেখাবো।

---

## ১. প্রজেক্ট সেটআপ

### প্রয়োজনীয় বিষয়সমূহ:
* Python 3.10+ (Python 3.11 সবচেয়ে ভালো)
* Git
* লোকাল ব্যবহারের জন্য [Ollama](https://ollama.com/)

### ভার্চুয়াল এনভায়রনমেন্ট তৈরি:
```bash
# প্রজেক্ট ক্লোন করুন
git clone https://github.com/paragmm/ParagAI.git
cd ParagAI

# ভার্চুয়াল এনভায়রনমেন্ট তৈরি করুন
python -m venv .venv

# উইন্ডোজে অ্যাক্টিভেট করুন:
.venv\Scripts\activate

# লিনাক্স বা ম্যাক-এ অ্যাক্টিভেট করুন:
source .venv/bin/activate

# ডেভ ডিপেনডেন্সিসহ ইনস্টল করুন
pip install -e ".[dev]"
```

---

## ২. প্রি-ফ্লাইট সিস্টেম ডায়াগনস্টিক (`doctor.py`)

সবকিছু ঠিক আছে কিনা তা যাচাই করতে ParagAI এর নিজস্ব ডায়াগনস্টিক টুল চালান:
```bash
python scripts/doctor.py
```

এটি আপনার পাইথন ভার্সন, CUDA জিপিইউ, VRAM, `torch.compile`, চেকপয়েন্ট ও টোকেনাইজার রাউন্ডট্রিপ এবং ওলামা সার্ভিস পরীক্ষা করে ফলাফল জানাবে।

---

## ৩. মডেল ট্রেনিং শুরু করা

ParagAI তে লাইভ ড্যাশবোর্ডসহ ট্রেনিং শুরু করতে:
```bash
python scripts/train.py \
  --data data/raw/chat/conversations_sft.txt \
  --config configs/train_config.yaml \
  --max_iters 500
```

টার্মিনালে লাইভ ড্যাশবোর্ড দেখা যাবে যা প্রতি সেকেন্ডে প্রসেস হওয়া টোকেন (`Tokens/sec`), VRAM ব্যবহার, লার্নিং রেট এবং ভ্যালিডেশন লস দেখাবে।

---

## ৪. টার্মিনালে চ্যাট করা

মডেল ট্রেনিং হওয়ার পর সরাসরি টার্মিনালে কথা বলতে চালান:
```bash
python scripts/chat.py --checkpoint checkpoints/best_model.pt --tokenizer checkpoints/tokenizer.json
```

---

## ৫. Ollama-তে ডিপ্লয়মেন্ট

আপনার মডেলকে লোকাল ওলামায় নিতে ৩টি সহজ ধাপ:

```bash
# ১. Hugging Face SafeTensors ফরম্যাটে রূপান্তর
python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output_dir checkpoints/export_hf --verify

# ২. GGUF ফাইল তৈরি (GPT-2 বাইট ম্যাপিংসহ)
python scripts/export_gguf.py --input_dir checkpoints/export_hf --tokenizer checkpoints/tokenizer.json --output checkpoints/paragai.gguf

# ৩. Ollama-তে রেজিস্টার করুন
ollama create paragai -f deploy/Modelfile

# ৪. সরাসরি ওলামায় চালান
ollama run paragai:latest "হ্যালো, একটি ছোট গল্প বলো"
```

---

## ৬. vLLM এ সার্ভার চালু করা

হাই-থ্রুপুট এন্টারপ্রাইজ API পরিবেশনের জন্য:
```bash
# লিনাক্সে:
bash scripts/serve_vllm.sh checkpoints/export_hf 8000

# উইন্ডোজে:
scripts\serve_vllm.bat checkpoints/export_hf 8000
```
OpenAI এর মতো সরাসরি `http://localhost:8000/v1/chat/completions` এন্ডপয়েন্টে রিকোয়েস্ট পাঠাতে পারবেন।
