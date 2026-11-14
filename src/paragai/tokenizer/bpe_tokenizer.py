"""Byte-Pair Encoding (BPE) subword tokenizer for ParagAI."""

import itertools
import json
from collections import defaultdict

from paragai.tokenizer.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """Subword Byte-Pair Encoding (BPE) Tokenizer."""

    def __init__(self, vocab_size: int = 512):
        self.target_vocab_size = vocab_size
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}

        # Initialize base 256 byte tokens after special tokens
        for idx, tok in enumerate(self.special_tokens):
            self.vocab[idx] = tok.encode("utf-8")

        for byte_val in range(256):
            self.vocab[len(self.vocab)] = bytes([byte_val])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
        counts = defaultdict(int)
        for p0, p1 in itertools.pairwise(ids):
            counts[(p0, p1)] += 1
        return counts

    @staticmethod
    def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text: str, verbose: bool = False) -> None:
        """Train BPE merge rules on raw text."""
        raw_bytes = list(text.encode("utf-8"))
        # Map raw bytes (0-255) to vocab IDs offset by special token count
        offset = len(self.special_tokens)
        ids = [b + offset for b in raw_bytes]

        num_merges = self.target_vocab_size - len(self.vocab)

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            new_id = len(self.vocab)

            ids = self.merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (i + 1) % 50 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> {new_id}")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        raw_bytes = list(text.encode("utf-8"))
        offset = len(self.special_tokens)
        ids = [b + offset for b in raw_bytes]

        while len(ids) >= 2:
            stats = self.get_stats(ids)
            # Find the lowest rank merge that exists in self.merges
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)

        if add_special_tokens:
            return [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        byte_parts = []
        for i in ids:
            if skip_special_tokens and i in (self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID):
                continue
            if i in self.vocab:
                byte_parts.append(self.vocab[i])
            else:
                byte_parts.append(self.UNK_TOKEN.encode("utf-8"))

        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def save(self, file_path: str) -> None:
        data = {
            "type": "bpe",
            "target_vocab_size": self.target_vocab_size,
            "merges": [
                {"pair": [int(p[0]), int(p[1])], "new_id": int(v)} for p, v in self.merges.items()
            ],
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "BPETokenizer":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(vocab_size=data["target_vocab_size"])
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.merges = {
            (int(m["pair"][0]), int(m["pair"][1])): int(m["new_id"]) for m in data["merges"]
        }
        return tok
