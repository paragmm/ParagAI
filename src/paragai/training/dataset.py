"""Dataset loaders and dynamic batching for autoregressive language modeling."""

import os
from typing import Tuple, Optional
import torch
import numpy as np
from paragai.tokenizer.base import BaseTokenizer


class TextDataset:
    """Memory-efficient text dataset with dynamic autoregressive batching."""

    def __init__(
        self,
        token_ids: np.ndarray,
        val_ratio: float = 0.1,
    ):
        self.data = token_ids
        split_idx = int(len(self.data) * (1.0 - val_ratio))
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: BaseTokenizer,
        val_ratio: float = 0.1,
    ) -> "TextDataset":
        """Tokenize raw text string into numpy array and create dataset."""
        tokens = tokenizer.encode(text)
        token_array = np.array(tokens, dtype=np.int32)
        return cls(token_array, val_ratio=val_ratio)

    @classmethod
    def from_file(
        cls,
        file_path: str,
        tokenizer: BaseTokenizer,
        val_ratio: float = 0.1,
    ) -> "TextDataset":
        """Load text from file and tokenize."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls.from_text(text, tokenizer, val_ratio=val_ratio)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of inputs X and targets Y where Y is X shifted by 1."""
        data = self.train_data if split == "train" else self.val_data
        if len(data) <= seq_len:
            raise ValueError(
                f"Data size ({len(data)}) must be strictly greater than seq_len ({seq_len})"
            )

        max_start = len(data) - seq_len - 1
        start_indices = np.random.randint(0, max_start + 1, size=batch_size)

        x_batch = np.stack([data[i : i + seq_len] for i in start_indices])
        y_batch = np.stack([data[i + 1 : i + seq_len + 1] for i in start_indices])

        x_tensor = torch.tensor(x_batch, dtype=torch.long, device=device)
        y_tensor = torch.tensor(y_batch, dtype=torch.long, device=device)

        return x_tensor, y_tensor

