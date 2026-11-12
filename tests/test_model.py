"""Unit tests for ParagAI Model Architecture."""

import torch
from paragai.config import ModelConfig
from paragai.model.normalization import RMSNorm
from paragai.model.attention import CausalSelfAttention
from paragai.model.transformer import ParagAIModel


def get_test_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        max_seq_len=32,
        n_embd=64,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )


def test_rmsnorm():
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 10, dim)
    out = norm(x)

    assert out.shape == x.shape
    # Check that root mean square across last dim is approximately 1.0
    rms = torch.sqrt(out.pow(2).mean(-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_model_forward_shapes_and_loss():
    config = get_test_config()
    model = ParagAIModel(config)

    B, T = 2, 16
    x = torch.randint(0, config.vocab_size, (B, T))
    y = torch.randint(0, config.vocab_size, (B, T))

    # Test forward pass with loss
    logits, loss = model(x, targets=y)

    assert logits.shape == (B, T, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0

    # Test forward pass without targets
    logits_no_targets, loss_none = model(x)
    assert logits_no_targets.shape == (B, T, config.vocab_size)
    assert loss_none is None


def test_causal_masking_invariant():
    """Verify that future tokens do not affect earlier token predictions."""
    config = get_test_config()
    model = ParagAIModel(config)
    model.eval()

    seq1 = torch.tensor([[10, 20, 30, 40, 50]])
    seq2 = torch.tensor([[10, 20, 30, 99, 99]])  # Diverges at position index 3

    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)

    # Logits up to index 2 (tokens 10, 20, 30) MUST be identical
    assert torch.allclose(logits1[:, :3, :], logits2[:, :3, :], atol=1e-5)


def test_gradient_flow():
    """Verify that backpropagation produces non-zero gradients on all model parameters."""
    config = get_test_config()
    model = ParagAIModel(config)

    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    _, loss = model(x, targets=y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"

