"""Unit tests for FastIndicBPETokenizer (Rust backend and fallback)."""

import json
import os
import tempfile

from paragai.tokenizer.fast_indic_bpe import FastIndicBPETokenizer


def test_fast_indic_bpe_initialization():
    tok = FastIndicBPETokenizer(vocab_size=4096)
    assert tok.target_vocab_size == 4096
    assert len(tok.special_tokens) == 8
    assert tok.special_tokens[0] == "<pad>"
    assert tok.special_tokens[1] == "<unk>"
    assert tok.special_tokens[2] == "<bos>"
    assert tok.special_tokens[3] == "<eos>"
    assert tok.special_tokens[4] == "<|im_start|>"
    assert tok.special_tokens[5] == "<|im_end|>"
    assert tok.special_tokens[6] == "<|endoftext|>"
    assert tok.special_tokens[7] == "<|tool_call|>"


def test_fast_indic_bpe_training_and_multilingual_roundtrip():
    tok = FastIndicBPETokenizer(vocab_size=4096)
    sample_corpus = (
        "Artificial intelligence and deep learning from first principles. "
        "কৃত্রিম বুদ্ধিমত্তা এবং বাংলা ভাষার মডেল। "
        "मशीन लर्निंग और प्राकृतिक भाषा प्रसंस्करण। "
        "செயற்கை நுண்ணறிவு மற்றும் தமிழ் மொழி. "
        "కృత్రిమ మేధస్సు మరియు తెలుగు భాష."
    )
    tok.train(sample_corpus, verbose=False)
    assert tok.vocab_size > 264

    test_sentences = [
        "Hello ParagAI, welcome to production!",
        "বাংলা ভাষায় প্রথম প্রিন্সিপলস থেকে এলএলএম তৈরি।",
        "नमस्ते भारत, यह एक आधुनिक मॉडल है।",
        "வணக்கம் நண்பா, எப்படி இருக்கிறீர்கள்?",
        "నమస్కారం, ఇది ప్రొఫెషనల్ టోకనైజర్.",
        "Math: 42 * 1337 = 56154 | Symbols: @#$%^&*",
    ]
    for s in test_sentences:
        encoded = tok.encode(s)
        decoded = tok.decode(encoded)
        assert decoded == s, f"Roundtrip failed for '{s}': got '{decoded}'"


def test_fast_indic_bpe_byte_fallback_emoji():
    tok = FastIndicBPETokenizer(vocab_size=4096)
    tok.train("Basic text without any emojis or rare symbols.", verbose=False)

    unseen_text = "Launching to production 🚀🌟 with high speed ⚡!"
    encoded = tok.encode(unseen_text)
    decoded = tok.decode(encoded)
    assert decoded == unseen_text


def test_fast_indic_bpe_save_and_load():
    tok = FastIndicBPETokenizer(vocab_size=4096)
    tok.train("Serialization test for FastIndicBPETokenizer. পরীক্ষা।", verbose=False)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        tok.save(tmp_path)
        assert os.path.exists(tmp_path)

        with open(tmp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "bpe"
        assert data["tokenizer_class"] == "fast_indic_bpe"

        loaded = FastIndicBPETokenizer.load(tmp_path)
        assert loaded.vocab_size == tok.vocab_size
        test_str = "Testing loaded tokenizer serialization পরীক্ষা"
        assert loaded.decode(loaded.encode(test_str)) == test_str
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_fast_indic_bpe_huggingface_export():
    tok = FastIndicBPETokenizer(vocab_size=4096)
    tok.train("HuggingFace export test text.", verbose=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        hf_path = tok.export_huggingface(tmp_dir)
        assert os.path.exists(hf_path)
        assert os.path.exists(os.path.join(tmp_dir, "tokenizer_config.json"))

        with open(os.path.join(tmp_dir, "tokenizer_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["model_type"] == "llama"
        assert cfg["bos_token"] == "<bos>"
        assert cfg["eos_token"] == "<eos>"


def test_fast_indic_bpe_coverage_stats():
    tok = FastIndicBPETokenizer(vocab_size=4096)
    text = "বাংলা এবং English text mixed together."
    tok.train(text, verbose=False)

    stats = tok.get_coverage_stats(text)
    assert stats["total_chars"] == len(text)
    assert stats["total_tokens"] > 0
    assert "Bengali" in stats["script"]
    assert "Latin" in stats["script"]
    assert stats["compression_ratio"] > 0.0
