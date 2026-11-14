# ParagAI Tutorial 06: Continuous Daily Training & Zero Knowledge-Loss Guide

This is the comprehensive master guide on **how to train ParagAI incrementally day-by-day** (e.g. 10–20 minutes daily) **without losing past knowledge** and without running into catastrophic forgetting or tokenizer corruption.

---

## 1. The Core Engineering Challenge: Catastrophic Forgetting

In machine learning, **Catastrophic Forgetting** occurs when a neural network trained on Task A (e.g., English fairy tales on Day 1) is subsequently trained *only* on Task B (e.g., Python code or Bengali conversations on Day 2). Without proper safeguards, the backpropagation updates for Task B overwrite the synaptic weights responsible for Task A. By Day 3, the model forgets how to tell stories.

Furthermore, **Tokenizers have strict architectural invariants**:
If you retrain `tokenizer.json` on Day 2 with new words, token IDs will shift (e.g., token ID 42 which meant `"garden"` on Day 1 might become `"function"` on Day 2). Because the neural network's embedding matrix `token_embedding.weight[42]` was already trained on the meaning of `"garden"`, shifting token IDs instantly corrupts the entire model's brain!

---

## 2. The 3 Golden Rules of Incremental Training in ParagAI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PARAGAI ZERO-LOSS ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. TOKENIZER FREEZE: Train IndicBPETokenizer ONCE with 256-Byte Fallback.    │
│    Never retrain tokenizer.json mid-project. New characters map to bytes!   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. REPLAY BUFFER (Experience Replay):                                       │
│    When training Day N data, always mix 15%-20% of Day 1...(N-1) data.       │
│    This preserves past synaptic pathways with minimal compute overhead!     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. WARM CHECKPOINT RESUME:                                                  │
│    Always resume from `checkpoints/best_model.pt` using lower learning rate │
│    with Cosine Warmup, so weights refine smoothly without sudden shocks.    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Daily Workflow

### Day 1: Base Pretraining (e.g., Stories & Foundational Text)
1. Place Day 1 data in `data/raw/stories/day1_stories.txt`.
2. Ensure your tokenizer has 256-byte fallback (`checkpoints/tokenizer_indic.json` or `checkpoints/tokenizer.json`).
3. Train the base checkpoint:
   ```bash
   python scripts/train.py \
     --data data/raw/stories/day1_stories.txt \
     --config configs/train_config.yaml \
     --max_iters 2000
   ```
4. Output: `checkpoints/best_model.pt` (Val Loss: ~0.028).

---

### Day 2: Adding New Capabilities (e.g., Bengali Conversations)
**Do NOT train on Day 2 data alone!**
Instead, use ParagAI's automated Replay Mixer (`scripts/prepare_tinystories_sft.py` or corpus merging):

```bash
# Combine Day 2 data with 15% replay of Day 1
python -c "
with open('data/raw/stories/day1_stories.txt', 'r', encoding='utf-8') as f1, \
     open('data/raw/chat/conversations_sft.txt', 'r', encoding='utf-8') as f2, \
     open('data/processed/day2_combined.txt', 'w', encoding='utf-8') as out:
    # 85% new conversation data + 15% replay story data
    out.write(f2.read() + '\n\n' + f1.read()[:50000])
"
```

Resume training from `checkpoints/best_model.pt`:
```bash
python scripts/train.py \
  --data data/processed/day2_combined.txt \
  --resume checkpoints/best_model.pt \
  --learning_rate 0.0001 \
  --max_iters 1000
```
**Result**: The model learns Bengali conversations while maintaining its English storytelling capabilities!

---

### Day 3: Adding Code & Tool Automation
1. Place tool/coding data in `data/raw/automation/tools_sft.txt`.
2. Interleave with a slice of the existing corpus:
   ```bash
   python -c "
   with open('data/processed/day2_combined.txt', 'r', encoding='utf-8') as prev, \
        open('data/raw/automation/tools_sft.txt', 'r', encoding='utf-8') as new_data, \
        open('data/processed/day3_combined.txt', 'w', encoding='utf-8') as out:
       out.write(new_data.read() + '\n\n' + prev.read()[:100000])
   "
   ```
3. Resume training:
   ```bash
   python scripts/train.py \
     --data data/processed/day3_combined.txt \
     --resume checkpoints/best_model.pt \
     --learning_rate 0.00008 \
     --max_iters 1000
   ```

---

## 4. What Happens When New Words or Characters Appear?

* **With Old CharTokenizer**: If a character was never in the training set, it became `<unk>` (unknown).
* **With ParagAI's `IndicBPETokenizer`**:
  Every unknown or newly encountered character decomposes cleanly into its **UTF-8 bytes (IDs 8–263)**.
  Because all 256 bytes exist in the vocabulary, the model can read, encode, and predict *any* Unicode character (even new Bengali ligatures, Hindi matras, emojis, or rare punctuation) **without modifying `tokenizer.json`**!

---

## 5. Verification Checklist After Each Daily Session

After each day's training run, run the automated verification suite to guarantee zero regression:

```bash
# 1. Run diagnostic self-check
python scripts/doctor.py

# 2. Run multi-task evaluation
python scripts/eval.py \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer checkpoints/tokenizer.json \
  --tasks multiling,tools,story

# 3. Export to Ollama for personal testing
python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output_dir checkpoints/export_hf
python scripts/export_gguf.py --input_dir checkpoints/export_hf --tokenizer checkpoints/tokenizer.json --output checkpoints/paragai.gguf
ollama create paragai -f deploy/Modelfile
ollama run paragai:latest "Hello! Tell me a story and show a tool command."
```
