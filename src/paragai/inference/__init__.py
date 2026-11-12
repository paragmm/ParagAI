"""ParagAI Inference and Generation."""

from paragai.inference.sampler import sample_next_token
from paragai.inference.generate import Generator

__all__ = ["sample_next_token", "Generator"]
