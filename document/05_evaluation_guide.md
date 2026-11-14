# ParagAI Tutorial 05: Benchmark Evaluation Suite

This guide explains how to quantitatively evaluate ParagAI models across perplexity, multilingual capabilities, tool use, and story generation using `scripts/eval.py`.

---

## 1. Overview of Evaluation Tasks

ParagAI includes a standardized benchmark evaluation suite (`scripts/eval.py`) covering 4 critical dimensions:

1. **Perplexity (`perplexity`)**:
   - Evaluates cross-entropy loss and perplexity ($PPL = e^{\mathcal{L}}$) on a held-out validation set.
   - Lower perplexity indicates superior predictive confidence and language mastery.

2. **Multilingual Proficiency (`multiling`)**:
   - Evaluates the model with 30 diverse prompts across English (10), Bengali (10), and Hindi (10).
   - Measures non-empty response rate and fluency without token crashes.

3. **Tool Adherence Test (`tools`)**:
   - Tests model generation on automation instructions (e.g. "Create a directory called logs", "List files in directory").
   - Measures the percentage of outputs producing syntactically valid JSON tool calls matching `{"action": "...", "command": "..."}`.

4. **Story Generation (`story`)**:
   - Tests narrative story generation against imaginative prompts.
   - Measures narrative structure adherence and story openings.

---

## 2. Running the Evaluation Suite

To evaluate your checkpoint:

```bash
python scripts/eval.py \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer checkpoints/tokenizer.json \
  --tasks multiling,tools,story
```

To include perplexity on validation data:
```bash
python scripts/eval.py \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer checkpoints/tokenizer.json \
  --val_data data/processed/combined_corpus.txt \
  --tasks perplexity,multiling,tools,story
```

---

## 3. Interpreting the Results Table

Example evaluation output:
```
============================================================
  Evaluation Results
============================================================

| Task             | Score                | Detail                                          |
|------------------|----------------------|-------------------------------------------------|
| Multilingual     | 100.0% non-empty     | english: 100.0%, bengali: 100.0%, hindi: 100.0% |
| Tool Adherence   | 60.0% valid JSON     | 3/5 prompts                                     |
| Story Generation | 80.0% narrative start | 4/5 prompts                                     |
| Perplexity       | 1.0287               | 12,450 tokens evaluated                         |

Results saved → checkpoints/eval_results.json
```

All metrics are automatically saved into `checkpoints/eval_results.json` so you can track your model's progress across days and iterations!
