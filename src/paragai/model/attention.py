"""Causal Multi-Head Self-Attention module."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from paragai.config import ModelConfig
from paragai.model.rope import apply_rope


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention with Rotary Position Embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

        # Projections for Q, K, V
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower triangular matrix of 1s
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V projections and reshape to (B, n_head, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE if frequencies are supplied
        if freqs_cis is not None:
            # Sliced to current sequence length T
            freqs = freqs_cis[:T]
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (mask out future tokens)
        causal_mask = self.mask[:, :, :T, :T]
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Context output: (B, n_head, T, head_dim)
        y = torch.matmul(weights, v)

        # Re-assemble all heads side by side: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection with residual dropout
        return self.resid_dropout(self.out_proj(y))

