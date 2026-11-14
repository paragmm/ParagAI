"""Causal Multi-Head Self-Attention module with PyTorch SDPA and KV-Cache."""

import torch
import torch.nn.functional as F
from torch import nn

from paragai.config import ModelConfig
from paragai.model.rope import apply_rope


class CausalSelfAttention(nn.Module):
    """
    Pro-Grade Causal Multi-Head Self-Attention.
    Supports PyTorch 2.0 SDPA (FlashAttention kernel acceleration) and Key-Value (KV) Caching.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.dropout_rate = config.dropout

        # Projections for Q, K, V
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = x.size()

        # Compute Q, K, V projections and reshape to (B, n_head, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Calculate position offset if KV-cache is present
        past_len = kv_cache[0].size(2) if kv_cache is not None else 0

        # Apply RoPE to query and key
        if freqs_cis is not None:
            freqs = freqs_cis[past_len : past_len + T]
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)

        # Update Key-Value cache
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        # Determine causal masking
        # When generating single token with past KV-cache, causal mask is not required
        is_causal = (T > 1) and (past_len == 0)

        # High-performance Scaled Dot-Product Attention (SDPA)
        # Automatically dispatches to FlashAttention-2 or memory-efficient kernels on GPU
        dropout_p = self.dropout_rate if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            dropout_p=dropout_p,
        )

        # Re-assemble all heads: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection with residual dropout
        out = self.resid_dropout(self.out_proj(y))
        return out, new_kv_cache
