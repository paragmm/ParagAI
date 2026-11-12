"""Strongly-typed configuration classes for ParagAI."""

from typing import Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for ParagAI Transformer Model."""
    name: str = "ParagAI-Small"
    vocab_size: int = Field(default=2048, ge=64, description="Vocabulary size")
    max_seq_len: int = Field(default=256, ge=16, description="Maximum context window length")
    n_embd: int = Field(default=192, ge=32, description="Embedding dimension")
    n_head: int = Field(default=6, ge=1, description="Number of attention heads")
    n_layer: int = Field(default=6, ge=1, description="Number of transformer layers")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate")
    bias: bool = Field(default=False, description="Whether to include bias in linear projections")
    rms_norm_eps: float = Field(default=1e-5, description="RMSNorm epsilon for numerical stability")
    rope_theta: float = Field(default=10000.0, description="Base frequency for RoPE")

    @property
    def head_dim(self) -> int:
        """Dimensionality of each attention head."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"Embedding dimension ({self.n_embd}) must be divisible by head count ({self.n_head})"
            )
        return self.n_embd // self.n_head


class TrainConfig(BaseModel):
    """Configuration for training ParagAI models."""
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=6e-4, gt=0.0)
    min_lr: float = Field(default=6e-5, ge=0.0)
    warmup_iters: int = Field(default=200, ge=0)
    max_iters: int = Field(default=5000, ge=1)
    weight_decay: float = Field(default=0.1, ge=0.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    eval_interval: int = Field(default=250, ge=1)
    eval_iters: int = Field(default=50, ge=1)
    device: str = Field(default="auto")
    checkpoint_dir: str = Field(default="checkpoints")
    data_dir: str = Field(default="data/processed")

