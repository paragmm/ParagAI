"""Full ParagAI Decoder-Only Transformer Language Model."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from paragai.config import ModelConfig
from paragai.model.normalization import RMSNorm
from paragai.model.rope import precompute_rope_freqs
from paragai.model.transformer_block import TransformerBlock


class ParagAIModel(nn.Module):
    """
    ParagAI Autoregressive Decoder-Only Transformer.
    Modern LLaMA-style architecture with RoPE, RMSNorm, SwiGLU, and weight tying.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # Precompute RoPE complex frequencies
        freqs_cis = precompute_rope_freqs(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis)

        # Stacked transformer decoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final normalization & language model head
        self.final_norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between token embedding and LM head
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights with standard Gaussian
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            idx: Tensor of token IDs with shape (Batch, SeqLen)
            targets: Optional tensor of target token IDs with shape (Batch, SeqLen)
        Returns:
            Tuple of (logits, loss). Loss is None if targets is not provided.
        """
        B, T = idx.size()
        assert T <= self.config.max_seq_len, (
            f"Input sequence length ({T}) exceeds max_seq_len ({self.config.max_seq_len})"
        )

        # Token embedding lookup
        x = self.token_embedding(idx)
        x = self.dropout(x)

        # Pass through transformer blocks with RoPE frequencies
        for block in self.blocks:
            x = block(x, freqs_cis=self.freqs_cis)

        # Final RMSNorm
        x = self.final_norm(x)

        # Project to vocabulary logits: (B, T, VocabSize)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten across batch and sequence dimensions for Cross-Entropy Loss
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

