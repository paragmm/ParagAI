# ParagAI টিউটোরিয়াল ০৩: ইন্ডিয়ান ও বহুভাষিক BPE টোকেনাইজার গাইড

এই গাইডে ParagAI-এর **IndicBPETokenizer**-এর গঠন, ট্রেইনিং পদ্ধতি এবং ইউনিকোড কভারেজ আলোচনা করা হয়েছে।

---

## ১. কেন ইন্ডিয়ান BPE টোকেনাইজার প্রয়োজন?

পাশ্চাত্যের স্ট্যান্ডার্ড টোকেনাইজারগুলোতে (যেমন GPT-2 বা LLaMA) বাংলা বা হিন্দি লেখার ক্ষেত্রে **টোকেন ফ্র্যাগমেন্টেশন (Token Fragmentation)** ঘটে। যেমন:
- `বাংলাদেশ` শব্দটি লিখতে GPT-2 টোকেনাইজারে ৬ থেকে ৮টি টোকেন লেগে যায়, কারণ প্রতিটি অক্ষর আলাদা বাইটে ভেঙে যায়।
- ফলে কনটেক্সট উইন্ডো দ্রুত শেষ হয়ে যায় এবং মডেল ধীরগতির হয়ে পড়ে।

ParagAI-এর `IndicBPETokenizer` এই সমস্যার সমাধান করে:
1. **বাংলা ও ভারতীয় লিপির সাব-ওয়ার্ড:** বাংলা যুক্তবর্ণ, স্বরচিহ্ন (`া`, `ি`, `ী`, `ে`, `্`) এবং হিন্দির যুক্তাক্ষর সরাসরি টোকেন হিসেবে সংরক্ষিত থাকে।
2. **১০০% বাইট ফলব্যাক (256-Byte Fallback):** সব ২৫৬টি UTF-8 বাইট আগে থেকেই ভোকেবুলারিতে থাকে। তাই যেকোনো নতুন শব্দ বা ইমোজির ক্ষেত্রে কখনো `<unk>` সমস্যা হয় না।
3. **ChatML স্পেশাল টোকেন:** `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, এবং `<|tool_call|>` শুরুতেই সংরক্ষিত।

---

## ২. টোকেনাইজার ট্রেইনিং করার নিয়ম

আপনার ডেটাসেট দিয়ে টোকেনাইজার ট্রেন করতে চালান:

```bash
python scripts/train_multilingual_bpe.py \
  --input "data/raw/**/*.txt" \
  --vocab_size 4096 \
  --output checkpoints/tokenizer_indic.json \
  --sample_mb 10
```

এটি স্বয়ংক্রিয়ভাবে বাংলা, হিন্দি এবং ইংরেজি অক্ষরের কম্প্রেশন রেশিও হিসাব করে এবং একটি `tokenizer_indic.json` ফাইল তৈরি করে।

---

## ৩. পাইথনে ব্যবহার করার কোড

```python
from paragai.tokenizer.indic_bpe import IndicBPETokenizer

tokenizer = IndicBPETokenizer.load("checkpoints/tokenizer_indic.json")
text = "বাংলা ভাষার মডেল 🚀"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
print(f"টোকেন সংখ্যা: {len(tokens)}")
assert decoded == text
```
