---
name: tokenizer-builder
description: >-
  Use this skill when implementing, training, evaluating, or debugging tokenizers (Character-level or
  Byte-Pair Encoding / BPE), managing vocabularies, or processing text corpora into token streams.
---

# ParagAI Tokenizer Builder Skill

This skill provides step-by-step procedures for building, training, and testing tokenizers for `ParagAI`.

## Available Tokenizer Types

1. **`BPETokenizer` (`src/paragai/tokenizer/bpe_tokenizer.py`) — [Mandatory for Pro-Grade]:**
   - Subword Byte-Pair Encoding algorithm (similar to GPT-2 / LLaMA).
   - Iteratively merges the most frequent byte/token pairs in the bilingual corpus.
   - Standard vocabulary size: 4,096 to 8,192 tokens.
   - Prevents character-level hallucinations and spelling errors.

2. **`CharTokenizer` (`src/paragai/tokenizer/char_tokenizer.py`) — [Prototype Only]:**
   - Maps each unique character to an integer ID.
   - Vocabulary size: 100 to 150 characters.
   - Strictly reserved for scalar engine testing and rapid overfitting smoke tests. Not suitable for production conversational intelligence.

---

## Tokenizer Standards & Special Tokens

Every tokenizer in ParagAI must register and reserve the following standard tokens:

| Token | ID | Meaning | Usage |
|---|---|---|---|
| `<pad>` | 0 | Padding | Used to pad short sequences in batched inference |
| `<unk>` | 1 | Unknown | Fallback for unseen characters/bytes |
| `<bos>` | 2 | Beginning of Sequence | Prepend to prompt inputs |
| `<eos>` | 3 | End of Sequence | Model emits this to stop generation |

---

## Standard Workflows

### 1. Training a BPE Tokenizer from Corpus
```python
from paragai.tokenizer.bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=2048)
tokenizer.train_from_file("data/raw/corpus.txt")
tokenizer.save("data/processed/tokenizer.json")
```

### 2. Validating Roundtrip Fidelity
Always verify that encoding followed by decoding perfectly reconstructs the original string:
```python
text = "ParagAI: Building AI models from first principles."
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
assert text == decoded, "Lossy tokenizer roundtrip detected!"
```

### 3. Running Unit Tests
```bash
pytest tests/test_tokenizer.py -v
```

