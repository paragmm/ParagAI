"""Tests for autoregressive sampling and generation."""

import torch
from paragai.config import ModelConfig
from paragai.model.transformer import ParagAIModel
from paragai.tokenizer.char_tokenizer import CharTokenizer
from paragai.inference.sampler import sample_next_token
from paragai.inference.generate import Generator


def test_greedy_sampler():
    logits = torch.tensor([1.0, 5.0, 2.0, -1.0])
    token = sample_next_token(logits, temperature=0.0)
    assert token == 1


def test_generator_output_length():
    text = "abcdefghijklmnopqrstuvwxyz"
    tokenizer = CharTokenizer(list(text))

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32,
        n_embd=32,
        n_head=2,
        n_layer=2,
    )
    model = ParagAIModel(model_config)
    generator = Generator(model, tokenizer, device=torch.device("cpu"))

    prompt = "abc"
    tokens = list(generator.stream_generate(prompt, max_new_tokens=10, temperature=0.8))
    assert len(tokens) == 10


def test_streaming_generator():
    text = "abcdefghijklmnopqrstuvwxyz"
    tokenizer = CharTokenizer(list(text))

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32,
        n_embd=32,
        n_head=2,
        n_layer=2,
    )
    model = ParagAIModel(model_config)
    generator = Generator(model, tokenizer, device=torch.device("cpu"))

    chunks = list(generator.stream_generate("ab", max_new_tokens=5, temperature=0.8))
    assert len(chunks) == 5
