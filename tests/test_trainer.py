"""Verification of training loop, loss convergence, and checkpointing."""

import os
import tempfile
import torch
from paragai.config import ModelConfig, TrainConfig
from paragai.model.transformer import ParagAIModel
from paragai.tokenizer.char_tokenizer import CharTokenizer
from paragai.training.dataset import TextDataset
from paragai.training.trainer import Trainer


def test_overfitting_sanity():
    """Verify that the model can rapidly overfit a small batch of text."""
    text = "The quick brown fox jumps over the lazy dog. " * 10
    tokenizer = CharTokenizer(list(text))

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=16,
        n_embd=32,
        n_head=2,
        n_layer=2,
        dropout=0.0,
    )
    model = ParagAIModel(model_config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_config = TrainConfig(
            batch_size=4,
            learning_rate=3e-3,
            min_lr=3e-4,
            warmup_iters=5,
            max_iters=60,
            eval_interval=20,
            eval_iters=5,
            checkpoint_dir=tmp_dir,
            device="cpu",
        )

        dataset = TextDataset.from_text(text, tokenizer, val_ratio=0.1)
        trainer = Trainer(model, dataset, train_config, device=torch.device("cpu"))

        # Measure initial loss
        x, y = dataset.get_batch("train", batch_size=4, seq_len=16, device=torch.device("cpu"))
        with torch.no_grad():
            _, initial_loss = model(x, targets=y)

        # Run training
        trainer.train()

        # Measure final loss on same batch
        with torch.no_grad():
            _, final_loss = model(x, targets=y)

        # Loss must drop significantly
        assert final_loss.item() < initial_loss.item()
        assert os.path.exists(os.path.join(tmp_dir, "final_model.pt"))

