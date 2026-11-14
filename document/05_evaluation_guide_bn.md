# ParagAI টিউটোরিয়াল ০৫: বেঞ্চমার্ক ইভালুয়েশন গাইড (Evaluation Guide)

এই গাইডে শেখানো হয়েছে কীভাবে `scripts/eval.py` দিয়ে আপনার ট্রেইন্ড মডেলের পারফরম্যান্স পরিমাপ করবেন।

---

## ১. ৪টি বেঞ্চমার্ক পরীক্ষা

ParagAI-তে ৪টি প্রধান মাপকাঠিতে মডেলকে টেস্ট করা হয়:

1. **পারপ্লেক্সিটি (`perplexity`):**
   - ভ্যালিডেশন ডেটার উপর মডেলের প্রেডিকশন লস ও পারপ্লেক্সিটি পরিমাপ করে। পারপ্লেক্সিটি যত কম হবে (যেমন 1.02 থেকে 2.50), মডেল তত নিখুঁত।

2. **বহুভাষিক বুদ্ধিমত্তা (`multiling`):**
   - ইংরেজি (১০টি), বাংলা (১০টি) এবং হিন্দি (১০টি) মোট ৩০টি প্রম্পটে মডেলকে টেস্ট করে। মডেল ঠিকমতো উত্তর দিচ্ছে কিনা তা যাচাই করে।

3. **টুল কলিং ও অটোমেশন (`tools`):**
   - শেল কমান্ড বা পাইথন কোড সংক্রান্ত নির্দেশ দিলে মডেল সঠিক JSON ফরম্যাটে (`{"action": "...", "command": "..."}`) আউটপুট দিতে পারছে কিনা তা টেস্ট করে।

4. **গল্প বলা (`story`):**
   - ক্রিয়েটিভ গল্প তৈরির প্রম্পটে মডেলের বর্ণনামূলক গুণমান যাচাই করে।

---

## ২. ইভালুয়েশন স্ক্রিপ্ট চালানোর কমান্ড

```bash
python scripts/eval.py \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer checkpoints/tokenizer.json \
  --tasks multiling,tools,story
```

ভ্যালিডেশন ডেটাসহ পারপ্লেক্সিটি মাপতে:
```bash
python scripts/eval.py \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer checkpoints/tokenizer.json \
  --val_data data/processed/combined_corpus.txt \
  --tasks perplexity,multiling,tools,story
```

ফলাফল স্বয়ংক্রিয়ভাবে একটি সুন্দর টেবিলে দেখানো হবে এবং `checkpoints/eval_results.json` ফাইলে সংরক্ষিত হবে।
