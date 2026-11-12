"""Tests for CharTokenizer and BPETokenizer roundtrip fidelity and serialization."""

import os
import tempfile
import torch
from paragai.tokenizer.char_tokenizer import CharTokenizer
from paragai.tokenizer.bpe_tokenizer import BPETokenizer
from paragai.training.dataset import TextDataset


def test_char_tokenizer_roundtrip():
    sample_text = "Hello ParagAI! Building a custom AI model from scratch."
    tokenizer = CharTokenizer(list(sample_text))

    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    assert decoded == sample_text
    assert tokenizer.vocab_size >= len(set(sample_text)) + 4


def test_char_tokenizer_serialization():
    sample_text = "Testing serialization."
    tokenizer = CharTokenizer(list(sample_text))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        tokenizer.save(temp_path)
        loaded = CharTokenizer.load(temp_path)
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.decode(loaded.encode(sample_text)) == sample_text
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_bpe_tokenizer_training_and_roundtrip():
    training_corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "paragai is an artificial intelligence model built from first principles. "
        "first principles thinking enables deep engineering understanding."
    )
    tokenizer = BPETokenizer(vocab_size=280)
    tokenizer.train(training_corpus)

    test_text = "the quick brown fox jumps over paragai"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    assert decoded == test_text
    assert tokenizer.vocab_size <= 280


def test_bpe_tokenizer_serialization():
    corpus = "Testing BPE tokenizer serialization and deserialization integrity."
    tokenizer = BPETokenizer(vocab_size=270)
    tokenizer.train(corpus)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        tokenizer.save(temp_path)
        loaded = BPETokenizer.load(temp_path)
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.decode(loaded.encode(corpus)) == corpus
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_dataset_batching():
    text = "abcdefghijklmnopqrstuvwxyz0123456789" * 10
    tokenizer = CharTokenizer(list(text))
    dataset = TextDataset.from_text(text, tokenizer, val_ratio=0.2)

    device = torch.device("cpu")
    B, T = 4, 8
    x, y = dataset.get_batch("train", batch_size=B, seq_len=T, device=device)

    assert x.shape == (B, T)
    assert y.shape == (B, T)
    # y should be shifted by 1 relative to x
    for b in range(B):
        assert torch.equal(x[b, 1:], y[b, :-1])

