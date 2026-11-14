"""ParagAI Inference and Generation."""

from paragai.inference.generate import Generator
from paragai.inference.sampler import sample_next_token

__all__ = ["Generator", "sample_next_token"]
