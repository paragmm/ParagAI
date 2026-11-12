"""SwiGLU Feed-Forward Network module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from paragai.config import ModelConfig


class SwiGLUFeedForward(nn.Module):
    """SwiGLU (Swish Gated Linear Unit) Feed-Forward Network."""

    def __init__(self, config: ModelConfig, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            # Standard LLaMA expansion: 4x n_embd
            hidden_dim = 4 * config.n_embd
            # Round to multiple of 32 or 64
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)

        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (SiLU(x * W_gate) * (x * W_up)) * W_down
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))

