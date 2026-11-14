"""Unit tests for IndicBPETokenizer and multilingual tokenization."""

import pytest
import os
import json
import tempfile
from paragai.tokenizer.indic_bpe import IndicBPETokenizer, _detect_scripts


def test_indic_bpe_initialization():
    tok = IndicBPETokenizer(vocab_size=4096)
    assert tok.target_vocab_size == 4096
    # Must have special tokens (8) + 256 bytes = 264 initial tokens
    assert tok.vocab_size == 264
    assert len(tok.special_tokens) == 8
    assert tok.special_tokens[4] == "<|im_start|>"
    assert tok.special_tokens[5] == "<|im_end|>"
    assert tok.special_tokens[6] == "<|endoftext|>"
    assert tok.special_tokens[7] == "<|tool_call|>"


def test_script_detection():
    assert "Latin" in _detect_scripts("Hello world")
    assert "Bengali" in _detect_scripts("হ্যালো বাংলা")
    assert "Devanagari" in _detect_scripts("नमस्ते हिंदी")


def test_indic_bpe_training_and_roundtrip():
    tok = IndicBPETokenizer(vocab_size=4096)
    text = (
        "Once upon a time in a sunny village. "
        "একটি ছোট মেয়ে বাগানে খেলছিল। "
        "एक छोटा लड़का बाग में खेल रहा था। "
    )
    tok.train(text, verbose=False)
    assert tok.vocab_size > 264

    # Test roundtrip fidelity
    for s in ["Hello world!", "ছোট মেয়ে", "छोटा लड़का", "123 + 456 = 579"]:
        encoded = tok.encode(s)
        decoded = tok.decode(encoded)
        assert decoded == s, f"Mismatch for '{s}': got '{decoded}'"


def test_indic_bpe_save_and_load():
    tok = IndicBPETokenizer(vocab_size=4096)
    text = "Machine learning and artificial intelligence. কৃত্রিম বুদ্ধিমত্তা।"
    tok.train(text, verbose=False)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        tok.save(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "bpe"
        assert data["tokenizer_class"] == "indic_bpe"

        loaded = IndicBPETokenizer.load(tmp_path)
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.decode(loaded.encode("Test testing কৃত্রিম")) == "Test testing কৃত্রিম"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_indic_bpe_byte_fallback_emoji():
    tok = IndicBPETokenizer(vocab_size=4096)
    # Emoji not in training data should still roundtrip due to 256-byte coverage
    emoji_str = "Rocket 🚀 Star ⭐"
    encoded = tok.encode(emoji_str)
    decoded = tok.decode(encoded)
    assert decoded == emoji_str
