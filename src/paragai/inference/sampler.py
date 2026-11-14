"""Probabilistic sampling methods for autoregressive generation."""

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    generated_tokens: torch.Tensor | None = None,
) -> int:
    """
    Sample the next token ID from output logits using temperature, top-k, and top-p (nucleus).
    logits shape: (VocabSize,)
    """
    # 1. Repetition penalty
    if repetition_penalty != 1.0 and generated_tokens is not None and len(generated_tokens) > 0:
        unique_tokens = torch.unique(generated_tokens)
        for token_id in unique_tokens:
            if logits[token_id] < 0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty

    # 2. Temperature scaling
    if temperature <= 0.0 or temperature < 1e-4:
        # Greedy argmax
        return torch.argmax(logits).item()

    logits = logits / temperature

    # 3. Top-k filtering
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[[-1]]] = float("-inf")

    # 4. Top-p (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right so at least one token remains
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    # 5. Softmax to probability distribution & multinomial sample
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token
