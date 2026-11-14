"""Residual Transformer Decoder Block with KV-Cache pass-through."""

import torch
from torch import nn

from paragai.config import ModelConfig
from paragai.model.attention import CausalSelfAttention
from paragai.model.feed_forward import SwiGLUFeedForward
from paragai.model.normalization import RMSNorm


class TransformerBlock(nn.Module):
    """Transformer Decoder Block with Pre-RMSNorm, SwiGLU, and KV-Cache."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Pre-norm architecture with residual stream
        attn_out, new_cache = self.attn(
            self.norm1(x),
            freqs_cis=freqs_cis,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache
