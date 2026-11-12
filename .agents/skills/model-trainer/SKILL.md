---
name: model-trainer
description: >-
  Use this skill when preparing datasets, configuring hyperparameters, running training loops,
  monitoring loss and perplexity, handling checkpoints, or debugging model convergence for ParagAI.
---

# ParagAI Model Trainer Skill

This skill governs the training pipeline, hyperparameter tuning, loss tracking, checkpoint management, and multi-task fine-tuning for `ParagAI`.

## 4-Stage Pro-Grade Training Execution Protocol

Always execute training in this strict sequence:

### Stage 1: Overfitting Sanity Check
Before running any long training session, verify the model can overfit a single small batch:
```bash
python scripts/train.py --config configs/sanity_test.yaml
```
- **Target:** Training loss must drop from $\approx 4.5$ to $< 0.1$ within 300 steps.
- **Troubleshooting:** If loss stalls at $\approx \ln(\text{vocab\_size})$, check learning rate, attention masks, and gradient zeroing.

### Stage 2: Foundation Pre-training (Autoregressive Next-Token)
Train base linguistic capabilities across bilingual text corpora:
```bash
python scripts/train.py --config configs/train_config.yaml
```
- **Target:** Validation loss $< 1.8$, perplexity $< 6.0$.
- **Validation Interval:** Evaluates every 250 iterations over 50 batches.
- **Checkpointing:** Saves to `checkpoints/best_model.pt` when validation loss reaches a new low.

### Stage 3: Supervised Fine-Tuning (SFT) for Dialogue & Multi-Task
Raw pre-training alone creates random text continuations. To achieve **Professional-Grade Chat & Task Execution**, the model must be fine-tuned on instruction datasets:
1. **ChatML Format:**
   ```text
   <|im_start|>user
   What are the core advantages of Transformers?<|im_end|>
   <|im_start|>assistant
   Transformers utilize self-attention mechanisms to process tokens in parallel...<|im_end|>
   ```
2. **Translation Conditioning (EN ↔ BN):**
   ```text
   <|im_start|>user
   [TASK: translate | source: en | target: bn]
   Artificial intelligence is transforming society.<|im_end|>
   <|im_start|>assistant
   কৃত্রিম বুদ্ধিমত্তা সমাজকে রূপান্তরিত করছে।<|im_end|>
   ```
3. **Agentic Tool Calling:**
   ```text
   <|im_start|>user
   [TASK: automation] List files in current directory.<|im_end|>
   <|im_start|>assistant
   <|tool_call|>{"action": "shell", "command": "dir"}<|im_end|>
   ```

### Stage 4: GPU Scaling on Google Colab / Cloud
For models $> 15\text{M}$ parameters with BPE tokenizers ($\ge 4,096$ vocab):
- Train using `notebooks/train_colab.ipynb` on NVIDIA T4/A100 GPUs.
- Batch size: 32 to 64, warmup steps: 500, AdamW with cosine decay.

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

