"""Residual Transformer Decoder Block."""

from typing import Optional
import torch
import torch.nn as nn
from paragai.config import ModelConfig
from paragai.model.normalization import RMSNorm
from paragai.model.attention import CausalSelfAttention
from paragai.model.feed_forward import SwiGLUFeedForward


class TransformerBlock(nn.Module):
    """Transformer Decoder Block with Pre-RMSNorm and Residual Connections."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFeedForward(config)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture with residual stream
        x = x + self.attn(self.norm1(x), freqs_cis=freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x

