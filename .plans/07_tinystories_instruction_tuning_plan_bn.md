# ParagAI ধাপে ধাপে TinyStories ইন্সট্রাকশন-টিউনিং মাস্টার প্ল্যান

**নথি আইডি:** `PLAN-07-BN`  
**উদ্দেশ্য:** ডেটা ইঞ্জিনিয়ারিং, স্বয়ংক্রিয় ইন্সট্রাকশন তৈরি, ধারাবাহিক SFT ট্রেনিং এবং Ollama ডিপ্লয়মেন্ট  
**প্রযোজ্য মডেল:** `ParagAI-Small` (১.৫M - ২.৫M) এবং `ParagAI-Medium` (১০M - ২৫M)  
**অবস্থা:** বাস্তবায়নের জন্য অনুমোদিত  

---

## ১. নির্বাহী সারসংক্ষেপ ও মূল লক্ষ্য

### ১.১ মূল সমস্যা
সাধারণভাবে কোনো কাঁচা গল্পের টেক্সট দিয়ে মডেল ট্রেন করালে সেটি একটি **Base Completion Model** (বাক্য সম্পূর্ণকারী) হয়ে ওঠে। তখন আপনি যদি তাকে নির্দেশ দেন:
```text
write a story for me
```
মডেলটি এটিকে নির্দেশ না ভেবে একটি অসম্পূর্ণ বাক্য মনে করে এবং এর পেছনে অবান্তর কিছু শব্দ জুড়ে দেয়, যেমন:
```text
...and take it to the teacher before the school bell rings.
```

### ১.২ সমাধান: ChatML ফরম্যাটে Instruction Tuning
ParagAI-কে নির্দেশ পালনকারী একনিষ্ঠ সহকারীতে রূপান্তর করতে মাইক্রোসফটের `roneneldan/TinyStories` ডেটাসেটের কাঁচা গল্পগুলোকে পাইথনের মাধ্যমে স্বয়ংক্রিয়ভাবে **ChatML ফরম্যাটে Instruction-Response জোড়া** হিসেবে সাজানো হবে:
```text
<|im_start|>system
You are ParagAI, an imaginative, helpful, and creative storytelling assistant.<|im_end|>
<|im_start|>user
Write a creative story for me.<|im_end|>
<|im_start|>assistant
Once upon a time, in a bright green meadow bordered by tall whispering pines...<|im_end|>
```

### ১.৩ প্রধান শর্ত ও টেকনিক্যাল নিয়মাবলি
1. **প্রতিদিন ধাপে ধাপে ট্রেনিং (Continual Learning):** এক সাথে ১০০ MB ডেটা প্রসেস না করে প্রতিদিন ৫ MB থেকে ১০ MB সাইজের ছোট অংশে ট্রেনিং হবে।
2. **টোকেনাইজার ফিক্সড রাখা (Fixed Vocabulary):** ট্রেনিং চলাকালীন `tokenizer_bpe.json` বা `tokenizer.json` কোনোভাবেই পরিবর্তন করা হবে না, যাতে PyTorch-এর এমবেডিং টেনসর ক্র্যাশ না করে।
3. **আগের জ্ঞান ধরে রাখা (Replay Buffer):** গল্পের সাথে সাথে প্রতিদিন ১০% পুরনো চ্যাট, অনুবাদ ও কোড ডেটা মিশিয়ে রাখা হবে, যাতে মডেল আগের শেখা বিষয়গুলো ভুলে না যায় (Catastrophic Forgetting রোধ)।

---

## ২. সামগ্রিক আর্কিটেকচার ও পাইপলাইন ডায়াগ্রাম

```
┌────────────────────────────────────────────────────────┐
│  পর্যায় ১: ডেটা সংগ্রহ (Data Acquisition)              │
│  Hugging Face থেকে roneneldan/TinyStories               │
│  প্রতিদিনের জন্য ৫ MB – ১০ MB স্লাইস নামানো              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  পর্যায় ২: স্বয়ংক্রিয় নির্দেশ তৈরি (Instruction Tuning)   │
│  scripts/prepare_tinystories_sft.py                    │
│  কাঁচা গল্প ──> ডাইনামিক প্রম্পট ──> ChatML SFT ডেটা   │
│  • গল্প লেখার সরাসরি নির্দেশ                            │
│  • চরিত্র ও পরিবেশভিত্তিক নির্দেশ                       │
│  • বাংলা ভাষায় গল্প লেখার প্রম্পট                       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  পর্যায় ৩: ধারাবাহিক ফাইন-টিউনিং (Sequential SFT)      │
│  scripts/train.py --resume checkpoints/best_model.pt   │
│  • কম লার্নিং রেট (1e-4 থেকে 5e-5)                    │
│  • প্রতিদিন ২,০০০ – ৩,০০০ স্টেপ                        │
│  • চেকবক্স সংরক্ষণ                                      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  পর্যায় ৪: GGUF এক্সপোর্ট ও Ollama ডিপ্লয়মেন্ট         │
│  SafeTensors ──> GGUF (GPT-2 Byte Mapping)             │
│  ollama create paragai -f deploy/Modelfile             │
│  যাচাইকরণ: "write a story for me" ──> সম্পূর্ণ নতুন গল্প│
└────────────────────────────────────────────────────────┘
```

---

## ৩. ধাপে ধাপে বাস্তবায়ন পরিকল্পনা

### ধাপ ০১: Hugging Face থেকে ছোট অংশে ডেটা নামানো
এক সাথে ১ GB ডেটা না নামিয়ে একটি স্ট্রিমিং পাইথন স্ক্রিপ্টের মাধ্যমে প্রথম দিন ৫ MB বা ১০ MB টেক্সট নামিয়ে নেওয়া হবে।

* **স্ক্রিপ্ট:** `scripts/fetch_tinystories.py`
* **সংরক্ষণ ঠিকানা:** `data/raw/stories/tinystories_chunk_01.txt`
* **সাইজ:** ~৫ MB থেকে ১০ MB (~১০,০০০ থেকে ২০,০০০ পরিচ্ছন্ন গল্প)।

---

### ধাপ ০২: কাঁচা গল্পকে নির্দেশ ও উত্তরের (Instruction) রূপ দেওয়া
মডেলকে *"write a story for me"* বা *"আমাকে একটি গল্প বলো"* শেখানোর জন্য প্রতিটি কাঁচা গল্পের সামনে স্বয়ংক্রিয়ভাবে বৈচিত্র্যময় প্রশ্ন ও প্রম্পট যোগ করা হবে।

* **স্ক্রিপ্ট:** `scripts/prepare_tinystories_sft.py`
* **আউটপুট ফাইল:** `data/processed/stories_sft_chunk_01.txt`

#### প্রম্পটের বিভিন্ন ধরন (Prompt Categories):
1. **সাধারণ গল্প লেখার নির্দেশ:**
   - *"Write a story for me."*
   - *"Please write an interesting and imaginative story."*
   - *"Can you tell me a story?"*
   - *"Compose a short story for me."*
2. **চরিত্রভিত্তিক নির্দেশ:**
   - গল্পের ভেতরের চরিত্র খুঁজে নিয়ে প্রম্পট তৈরি: *"Write a children's story about Lily and a magic key."*
3. **ঘুমপাড়ানি বা শান্ত গল্পের প্রম্পট:**
   - *"Tell me a calming bedtime story."*
   - *"Write a sweet story suitable for kids."*
4. **বাংলা ভাষার প্রম্পট:**
   - *"আমাকে একটি সুন্দর রূপকথার গল্প বলো।"*
   - *"আমার জন্য একটি শিক্ষণীয় গল্প লিখো।"*

#### ChatML ফরম্যাটের কাঠামো:
```text
<|im_start|>system
You are ParagAI, a creative and imaginative AI storytelling assistant.<|im_end|>
<|im_start|>user
Write a story for me.<|im_end|>
<|im_start|>assistant
Once upon a time, there was a little bird named Pip. Pip had bright blue wings and loved to sing at sunrise...<|im_end|>
```

---

### ধাপ ০৩: টোকেনাইজার সুরক্ষা ও অপরিবর্তনশীল রাখা
* **নিয়ম:** ধারাবাহিক ফাইন-টিউনিংয়ের সময় `tokenizer.json` বা `tokenizer_bpe.json` ফাইল কোনোভাবেই রি-ট্রেন বা এডিট করা যাবে না।
* **বাইটের সুরক্ষা (Byte-Fallback):** গল্পে যদি এমন কোনো নতুন প্রতীক বা বিরল বর্ণ থাকে যা আগে দেখা যায়নি, আমাদের BPE টোকেনাইজার সেটিকে তার অভ্যন্তরীণ ২৫৬টি বাইট দিয়ে রূপান্তর করে নেবে। কোনো এরর আসবে না।

---

### ধাপ ০৪: ধারাবাহিক ট্রেনিং প্রটোকল (Sequential Training)

প্রতিদিন ট্রেনিংয়ের সময় হাইপারপ্যারামিটারগুলো এভাবে সেট করা হবে:

| প্যারামিটার | প্রাথমিক প্রি-ট্রেনিং | ১ম দিনের SFT | ২য় দিন ও পরবর্তী দিনগুলো |
| :--- | :--- | :--- | :--- |
| **Resume Flag** | `None` | `--resume checkpoints/best_model.pt` | `--resume checkpoints/best_model.pt` |
| **Max Iterations** | ৫,০০০ | ২,৫০০ - ৩,০০০ | ২,০০০ - ২,৫০০ |
| **Learning Rate** | $6.0 \times 10^{-4}$ | $2.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ |
| **Min Learning Rate** | $6.0 \times 10^{-5}$ | $2.0 \times 10^{-5}$ | $1.0 \times 10^{-5}$ |
| **Warmup Steps** | ২০০ | ৫০ | ৫০ |
| **Batch Size** | ৩২ | ৩২ | ৩২ |
| **Sequence Length** | ২৫৬ | ২৫৬ | ২৫৬ |

#### টার্মিনালে রান করার কমান্ড:
```powershell
python scripts/train.py `
    --resume checkpoints/best_model.pt `
    --data data/processed/stories_sft_chunk_01.txt `
    --config configs/train_config.yaml
```

---

### ধাপ ০৫: পুরনো জ্ঞান ভুলে যাওয়া রোধ (Replay Mix)
মডেল যেন শুধু গল্প শিখতে গিয়ে আগের চ্যাট, অনুবাদ বা টুল কলিং ভুলে না যায়, সেজন্য ডেটা তৈরির স্ক্রিপ্ট প্রতিটি নতুন ৫ MB ডেটায় পুরনো ডেটা মিশিয়ে দেবে:
* **৮৫%** নতুন TinyStories ডেটা।
* **৫%** সাধারণ কথোপকথন (`conversations_sft.txt`)।
* **৫%** বাংলা-ইংরেজি অনুবাদ (`parallel_sft.txt`)।
* **৫%** স্বয়ংক্রিয় কমান্ড ও টুলস (`tools_sft.txt`)।

---

### ধাপ ০৬: পরীক্ষা ও ফলাফল যাচাই

#### টার্মিনাল টেস্ট:
[scripts/chat.py](file:///c:/Users/hedpa/Documents/AI/ParagAI/scripts/chat.py) চালিয়ে নিচের প্রম্পটগুলো দিয়ে টেস্ট করা হবে:
```text
Prompt: write a story for me
Prompt: Tell me a story about a little bear
Prompt: আমাকে একটি রূপকথার গল্প শোনাও
```

**সফলতার মানদণ্ড:**
1. মডেল প্রম্পটের সাথে উল্টাপাল্টা বাক্য না জুড়ে দিয়ে সরাসরি গল্পের শুরুতে চলে যাবে (*"Once upon a time..."* বা *"একদা এক দেশে..."*)।
2. গল্পটি শুরু, মাঝের ঘটনা এবং সমাপ্তি নিয়ে সুসংগত হবে।
3. গল্প শেষ হলে `<|im_end|>` দ্বারা সুন্দরভাবে থেমে যাবে।

---

### ধাপ ০৭: GGUF এক্সপোর্ট ও Ollama-তে সরাসরি রান

ট্রেনিং শেষ হওয়ার পর:
1. **SafeTensors এক্সপোর্ট:**
   ```powershell
   python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output_dir checkpoints/export_hf
   ```
2. **GGUF তৈরি:**
   ```powershell
   python scripts/export_gguf.py --input_dir checkpoints/export_hf --tokenizer checkpoints/tokenizer.json --output checkpoints/paragai.gguf
   ```
3. **Ollama রেজিস্ট্রেশন:**
   ```powershell
   ollama create paragai -f deploy/Modelfile
   ```
4. **Ollama টেস্ট:**
   ```powershell
   ollama run paragai:latest "write a story for me"
   ```

---

## ৪. প্রতিদিনের ট্রেনিং রুটিন ক্যালেন্ডার

| দিন | ডেটাসেটের অংশ | সাইজ | স্টেপ সংখ্যা | অর্জিত নতুন দক্ষতা |
| :--- | :--- | :--- | :--- | :--- |
| **১ম দিন** | `tinystories_sft_chunk_01.txt` | ৫ MB | ২,৫০০ | *"write a story for me"* বুঝতে পারা এবং সরল গল্প বলা। |
| **২য় দিন** | `tinystories_sft_chunk_02.txt` | ৫ MB | ২,৫০০ | চরিত্রের সংলাপ এবং সুন্দর পরিবেশের বর্ণনা দেওয়া। |
| **৩য় দিন** | `tinystories_sft_chunk_03.txt` | ৮ MB | ৩,০০০ | গল্পের সমস্যা ও সমাধান (Conflict & Resolution) ফুটিয়ে তোলা। |
| **৪র্থ দিন**| `bengali_stories_sft_chunk_01.txt` | ৫ MB | ২,৫০০ | সাবলীল প্রমিত বাংলায় গল্প বানিয়ে বলা। |
| **৫ম+ দিন**| ১০ MB প্রতি ব্যাচ | ১০ MB | ৩,০০০ | দীর্ঘ ও বৈচিত্র্যময় বড় বড় গল্প স্বতঃস্ফূর্তভাবে লেখা। |

