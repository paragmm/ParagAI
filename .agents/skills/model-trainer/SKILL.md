---
name: model-trainer
description: >-
  Use this skill when preparing datasets, configuring hyperparameters, running training loops,
  monitoring loss and perplexity, handling checkpoints, or debugging model convergence for ParagAI.
---

# ParagAI Model Trainer Skill

This skill governs the training pipeline, hyperparameter tuning, loss tracking, and checkpoint management for `ParagAI`.

## 3-Stage Training Execution Protocol

Always execute training in this strict sequence:

### Stage 1: Overfitting Sanity Check
Before running any long training session, verify the model can overfit a single small batch:
```bash
python scripts/train.py --config configs/sanity_test.yaml
```
- **Target:** Training loss must drop from $\approx 4.5$ to $< 0.1$ within 300 steps.
- **Troubleshooting:** If loss stalls at $\approx \ln(\text{vocab\_size})$, check learning rate, attention masks, and gradient zeroing.

### Stage 2: Baseline Training (TinyShakespeare)
```bash
python scripts/prepare_data.py --dataset tinyshakespeare
python scripts/train.py --config configs/train_config.yaml
```
- **Target:** Validation loss $< 1.6$, perplexity $< 5.0$.
- **Validation Interval:** Evaluates every 250 iterations over 50 batches.
- **Checkpointing:** Saves to `checkpoints/best_model.pt` when validation loss reaches a new low.

### Stage 3: Extended / Custom Training
For custom text or large corpora (e.g., TinyStories, Bengali corpus):
```bash
python scripts/train.py --config configs/train_config.yaml --data data/raw/custom.txt --resume checkpoints/latest.pt
```

---

## Convergence & Metric Diagnostics

| Symptom | Probable Cause | Action |
|---|---|---|
| **Loss is `NaN` or `Inf`** | Exploding gradients or excessive learning rate | Ensure `grad_clip = 1.0` is active; lower learning rate by $2\times$. |
| **Validation Loss rises while Train Loss falls** | Overfitting | Increase dropout from `0.1` to `0.2`, increase weight decay, or expand dataset. |
| **Loss oscillates erratically** | Batch size too small or LR schedule too aggressive | Increase `batch_size` (e.g. 16 $\to$ 32), lengthen warmup steps. |
| **Out Of Memory (OOM)** | Context length or batch size exceeds RAM/VRAM | Lower `batch_size` or reduce `max_seq_len`. |

---

## Verification Commands
```bash
# Check latest checkpoint
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Interactive smoke test
python scripts/chat.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"
```

