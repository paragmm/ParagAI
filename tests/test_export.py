"""Tests for SafeTensors and GGUF export pipelines."""

import os
import tempfile
import torch
from paragai.config import ModelConfig
from paragai.model.transformer import ParagAIModel
from paragai.tokenizer.char_tokenizer import CharTokenizer
from scripts.export_hf import convert_checkpoint_to_hf
from scripts.export_gguf import convert_to_gguf


def test_hf_and_gguf_export():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Create a dummy tokenizer & model
        text = "Hello world! ParagAI export test."
        tokenizer = CharTokenizer(list(text))
        tok_path = os.path.join(tmp_dir, "tokenizer.json")
        tokenizer.save(tok_path)

        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=16,
            n_embd=32,
            n_head=2,
            n_layer=2,
        )
        model = ParagAIModel(config)

        # 2. Save PyTorch checkpoint
        ckpt_path = os.path.join(tmp_dir, "model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": config.model_dump(),
        }, ckpt_path)

        # 3. Export to HuggingFace SafeTensors
        hf_dir = os.path.join(tmp_dir, "export_hf")
        convert_checkpoint_to_hf(ckpt_path, hf_dir)

        assert os.path.exists(os.path.join(hf_dir, "model.safetensors"))
        assert os.path.exists(os.path.join(hf_dir, "config.json"))

        # 4. Convert to GGUF
        gguf_path = os.path.join(tmp_dir, "paragai.gguf")
        convert_to_gguf(hf_dir, tok_path, gguf_path)

        assert os.path.exists(gguf_path)
        assert os.path.getsize(gguf_path) > 0

