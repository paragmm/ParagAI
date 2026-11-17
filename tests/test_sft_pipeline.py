"""Unit tests for ParagAI SFT Pipeline: Special Tokens, Loss Masking, Embedding Resize, and Generation Stop."""

import unittest
import torch
import numpy as np

from paragai.config import ModelConfig
from paragai.model.transformer import ParagAIModel
from paragai.tokenizer.base import BaseTokenizer
from paragai.tokenizer.fast_indic_bpe import FastIndicBPETokenizer
from paragai.training.dataset import SFTDataset
from paragai.inference.generate import Generator, format_chat_prompt


class TestSFTPipeline(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            vocab_size=4096,
            n_embd=64,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            max_seq_len=128,
            dropout=0.0,
        )
        self.model = ParagAIModel(self.config)

    def test_special_tokens_definitions(self):
        """Verify that standard special tokens exist on BaseTokenizer."""
        self.assertEqual(BaseTokenizer.SYSTEM_TOKEN, "<|system|>")
        self.assertEqual(BaseTokenizer.USER_TOKEN, "<|user|>")
        self.assertEqual(BaseTokenizer.ASSISTANT_TOKEN, "<|assistant|>")
        self.assertEqual(BaseTokenizer.END_TOKEN, "<|end|>")

    def test_resize_token_embeddings(self):
        """Verify resize_token_embeddings updates shapes, tied weights, and config."""
        orig_vocab = self.model.config.vocab_size
        new_vocab = orig_vocab + 4

        # Save an original weight slice to check preservation
        orig_weight_slice = self.model.token_embedding.weight[:10, :].clone()

        self.model.resize_token_embeddings(new_vocab)
        self.assertEqual(self.model.config.vocab_size, new_vocab)
        self.assertEqual(self.model.token_embedding.weight.shape, (new_vocab, self.config.n_embd))
        self.assertEqual(self.model.lm_head.weight.shape, (new_vocab, self.config.n_embd))

        # Check weight tying
        self.assertTrue(self.model.token_embedding.weight is self.model.lm_head.weight)

        # Check preservation of existing embeddings
        torch.testing.assert_close(self.model.token_embedding.weight[:10, :], orig_weight_slice)

    def test_cross_entropy_ignore_index_minus_100(self):
        """Verify that F.cross_entropy ignores targets with value -100."""
        batch_size = 2
        seq_len = 8
        idx = torch.randint(0, self.model.config.vocab_size, (batch_size, seq_len))

        # Target where all tokens are -100 except one
        targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        targets[0, 2] = 42

        logits, loss = self.model(idx, targets=targets)
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_sft_dataset_loss_masking(self):
        """Verify SFTDataset masks prompt tokens with -100 and supervises assistant tokens."""
        # Create a mock tokenizer with known token-to-id mapping
        class MockTokenizer(BaseTokenizer):
            PAD_ID = 0
            UNK_ID = 1
            BOS_ID = 2
            EOS_ID = 3
            SYSTEM_ID = 4
            USER_ID = 5
            ASSISTANT_ID = 6
            END_ID = 7

            def __init__(self):
                self.mapping = {
                    "<pad>": 0,
                    "<unk>": 1,
                    "<bos>": 2,
                    "<eos>": 3,
                    "<|system|>": 4,
                    "<|user|>": 5,
                    "<|assistant|>": 6,
                    "<|end|>": 7,
                    "Sys": 10,
                    "Hello": 20,
                    "Hi": 30,
                    "there": 31,
                }
                self.rev_mapping = {v: k for k, v in self.mapping.items()}

            def encode(self, text, add_special_tokens=False):
                # Simple split on tokens
                tokens = []
                for part in text.replace("\n", " ").split():
                    if part in self.mapping:
                        tokens.append(self.mapping[part])
                return tokens

            def decode(self, ids, skip_special_tokens=False):
                return " ".join([self.rev_mapping.get(i, "") for i in ids])

            @property
            def vocab_size(self):
                return 100

            def save(self, p): pass
            @classmethod
            def load(cls, p): return cls()

        mock_tok = MockTokenizer()
        dialogue = "<|system|> Sys <|end|>\n<|user|> Hello <|end|>\n<|assistant|> Hi there <|end|>"

        dataset = SFTDataset.from_text(dialogue, tokenizer=mock_tok, val_ratio=0.0)
        self.assertGreater(len(dataset.samples), 0)

        x, y = dataset.samples[0]
        # X: [<|system|>, Sys, <|end|>, <|user|>, Hello, <|end|>, <|assistant|>, Hi, there]
        # Y: [       -100, -100,  -100,   -100,   -100,   -100,            Hi, there, <|end|>]
        # Targets for system and user must be -100
        # Target for "Hi" (30), "there" (31), and "<|end|>" (7) must NOT be -100
        self.assertEqual(y[0], -100) # Sys
        self.assertEqual(y[1], -100) # <|end|>
        self.assertEqual(y[2], -100) # <|user|>
        self.assertEqual(y[3], -100) # Hello
        self.assertEqual(y[4], -100) # <|end|>
        self.assertEqual(y[5], -100) # <|assistant|>

        # Assistant tokens must have valid targets
        self.assertEqual(y[6], 30) # Hi
        self.assertEqual(y[7], 31) # there
        self.assertEqual(y[8], 7)  # <|end|>

    def test_format_chat_prompt(self):
        """Verify format_chat_prompt adheres to standard ParagAI template."""
        prompt = format_chat_prompt("What is 2+2?", system_message="Be concise.")
        expected = "<|system|>Be concise.<|end|>\n<|user|>What is 2+2?<|end|>\n<|assistant|>"
        self.assertEqual(prompt, expected)

    def test_generator_stop_on_end_token(self):
        """Verify that Generator halts when <|end|> token is predicted."""
        class MockStopTokenizer(BaseTokenizer):
            PAD_ID = 0
            UNK_ID = 1
            BOS_ID = 2
            EOS_ID = 3
            END_ID = 7

            def encode(self, text, add_special_tokens=False):
                return [10, 11]

            def decode(self, ids, skip_special_tokens=False):
                return "".join(str(i) for i in ids)

            def token_to_id(self, token):
                if token == "<|end|>":
                    return 7
                return None

            @property
            def vocab_size(self):
                return 50

            def save(self, p): pass
            @classmethod
            def load(cls, p): return cls()

        tok = MockStopTokenizer()
        # Mock model whose lm_head always emits logits favoring END_ID (7)
        mock_model = ParagAIModel(ModelConfig(vocab_size=50, n_embd=32, n_layer=1, n_head=1, max_seq_len=64))
        generator = Generator(mock_model, tok)

        # Force model lm_head weight so token 7 has massive logit
        with torch.no_grad():
            mock_model.lm_head.weight[7, :] = 1000.0

        out = generator.generate("dummy prompt", max_new_tokens=50, stop_on_eos=True, include_prompt=False)
        # Should stop after generating at most 1 token (token 7), returning empty or single token
        self.assertLessEqual(len(out), 5)


if __name__ == "__main__":
    unittest.main()
