"""Rotary Position Embedding (RoPE) implementation."""

import torch


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex frequencies for Rotary Position Embeddings."""
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
    # dim/2 frequencies
    theta_numerator = torch.arange(0, head_dim, 2).float()
    freqs = 1.0 / (theta ** (theta_numerator / head_dim))

    # Positions 0 to max_seq_len - 1
    t = torch.arange(max_seq_len).float()
    # Outer product: (max_seq_len, head_dim / 2)
    freqs = torch.outer(t, freqs)
    # Convert to polar form on unit circle: cos + i * sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to query or key tensor.
    x shape: (batch_size, n_head, seq_len, head_dim)
    freqs_cis shape: (seq_len, head_dim // 2)
    """
    # Reshape x into pairs of real coordinates -> view as complex numbers
    # (B, H, T, D) -> (B, H, T, D/2, 2) -> (B, H, T, D/2) as complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Match dimensions for broadcasting: (1, 1, seq_len, head_dim // 2)
    freqs_cis = freqs_cis.view(1, 1, x.shape[2], -1)

    # Complex multiplication rotates the vector
    x_rotated = x_complex * freqs_cis

    # Convert back to real coordinates
    x_out = torch.view_as_real(x_rotated).flatten(3)
    return x_out.type_as(x)
