# ParagAI Tutorial 03: Indic + Multilingual BPE Tokenizer Guide

This guide explains the architecture, training procedure, and Unicode coverage mechanics of ParagAI's **IndicBPETokenizer**.

---

## 1. Why Indic BPE Tokenization?

Standard Western tokenizers (such as GPT-2, GPT-4, and base LLaMA) have high **token fragmentation** on Indian scripts. For example:
- A Bengali word like `ভালোবাসা` often requires **6 to 10 tokens** in GPT-2 because each UTF-8 byte is split into a separate token.
- This inflates the context window, drastically slows down generation, and degrades reasoning capability.

ParagAI's `IndicBPETokenizer` solves this with:
1. **Configurable vocabulary sizes**: 4,096, 8,192, 16,384, or 32,768 tokens.
2. **Native Indic Unicode Subwords**: Bengali (যুক্তবর্ণ, স্বরচিহ্ন), Devanagari (मात्राएं, संयुक्त अक्षर), Tamil, and Telugu syllables.
3. **Guaranteed 256-Byte Fallback**: All 256 possible UTF-8 byte values are pre-allocated in the initial vocabulary. Zero out-of-vocabulary (`<unk>`) errors will ever occur for unseen characters, emojis, or code syntax!
4. **ChatML Special Tokens**: Pre-registered IDs for `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, and `<|tool_call|>`.

---

## 2. Vocabulary Structure

```
Index 0–3:    Base Special Tokens (<pad>=0, <unk>=1, <bos>=2, <eos>=3)
Index 4–7:    ChatML Special Tokens (<|im_start|>=4, <|im_end|>=5, <|endoftext|>=6, <|tool_call|>=7)
Index 8–263:  Complete 256 UTF-8 byte fallback (b'\x00' through b'\xff')
Index 264+:   Learned BPE merges (English, Bengali, Hindi, code, punctuation)
```

---

## 3. Training a Multilingual Tokenizer

To train an `IndicBPETokenizer` on your corpus:

```bash
python scripts/train_multilingual_bpe.py \
  --input "data/raw/**/*.txt" \
  --vocab_size 4096 \
  --output checkpoints/tokenizer_indic.json \
  --sample_mb 10
```

The script reports:
- Detected Unicode scripts (Latin, Bengali, Devanagari, Tamil, Telugu).
- Total characters processed.
- Compression ratio (characters per token) for each script.
- Lossless roundtrip verification on sample sentences.

---

## 4. Using the Tokenizer in Python

```python
from paragai.tokenizer.indic_bpe import IndicBPETokenizer

# Load trained tokenizer
tokenizer = IndicBPETokenizer.load("checkpoints/tokenizer_indic.json")

# Encode text
text = "Hello বিশ্ব! नमस्ते दुनिया! 🚀"
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")

# Lossless decode
decoded = tokenizer.decode(token_ids)
assert decoded == text
print("Roundtrip successful!")
```
