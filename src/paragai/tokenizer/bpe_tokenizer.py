"""Byte-Pair Encoding (BPE) subword tokenizer for ParagAI."""

from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict

from paragai.tokenizer.base import BaseTokenizer

try:
    import regex as re

    _WORD_REGEX = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    import re  # type: ignore[no-redef]

    _WORD_REGEX = re.compile(r"""\s+|[^\s]+""")


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
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for p0, p1 in itertools.pairwise(ids):
            counts[(p0, p1)] += 1
        return counts

    @staticmethod
    def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        new_ids: list[int] = []
        i = 0
        p0, p1 = pair
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == p0 and ids[i + 1] == p1:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text: str, verbose: bool = False) -> None:
        """Train BPE merge rules on raw text using high-throughput word frequency hashing.

        Parameters
        ----------
        text:
            Input corpus string.
        verbose:
            If True, prints progress every 50 merges.
        """
        if not text:
            return

        words = _WORD_REGEX.findall(text)
        if not words:
            return

        word_freqs = Counter(words)
        offset = len(self.special_tokens)

        # Word representation: word_index -> list of token IDs
        word_list: list[list[int]] = []
        word_counts: list[int] = []
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)

        for w_idx, (word, count) in enumerate(word_freqs.items()):
            raw_bytes = list(word.encode("utf-8"))
            token_ids = [b + offset for b in raw_bytes]
            word_list.append(token_ids)
            word_counts.append(count)
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                pair_counts[pair] += count
                pair_to_words[pair].add(w_idx)

        num_merges = self.target_vocab_size - len(self.vocab)
        if num_merges <= 0:
            return

        for m in range(num_merges):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] <= 0:
                break

            new_id = len(self.vocab)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            affected_words = list(pair_to_words[best_pair])
            p0, p1 = best_pair

            for w_idx in affected_words:
                old_tokens = word_list[w_idx]
                count = word_counts[w_idx]

                # Decrement old adjacent pairs
                for i in range(len(old_tokens) - 1):
                    p = (old_tokens[i], old_tokens[i + 1])
                    pair_counts[p] -= count
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    pair_to_words[p].discard(w_idx)
                    if not pair_to_words[p]:
                        del pair_to_words[p]

                # Merge occurrences in the word
                new_tokens: list[int] = []
                i = 0
                while i < len(old_tokens):
                    if i < len(old_tokens) - 1 and old_tokens[i] == p0 and old_tokens[i + 1] == p1:
                        new_tokens.append(new_id)
                        i += 2
                    else:
                        new_tokens.append(old_tokens[i])
                        i += 1

                # Increment new adjacent pairs
                for i in range(len(new_tokens) - 1):
                    p = (new_tokens[i], new_tokens[i + 1])
                    pair_counts[p] += count
                    pair_to_words[p].add(w_idx)

                word_list[w_idx] = new_tokens

            if verbose and (m + 1) % 50 == 0:
                print(f"Merge {m + 1}/{num_merges}: {best_pair} -> {new_id}")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return [self.BOS_ID, self.EOS_ID] if add_special_tokens else []

        # For long multi-word texts, pre-tokenize words to keep O(len(word)^2) bounded
        words = _WORD_REGEX.findall(text)
        offset = len(self.special_tokens)
        all_ids: list[int] = []

        for word in words:
            raw_bytes = list(word.encode("utf-8"))
            ids = [b + offset for b in raw_bytes]

            while len(ids) >= 2:
                stats = self.get_stats(ids)
                pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                ids = self.merge(ids, pair, idx)

            all_ids.extend(ids)

        if add_special_tokens:
            return [self.BOS_ID] + all_ids + [self.EOS_ID]
        return all_ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        byte_parts: list[bytes] = []
        special_ids = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}
        for i in ids:
            if skip_special_tokens and i in special_ids:
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
                {"pair": [int(p[0]), int(p[1])], "new_id": int(v)}
                for p, v in self.merges.items()
            ],
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str) -> BPETokenizer:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(vocab_size=data["target_vocab_size"])
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.merges = {
            (int(m["pair"][0]), int(m["pair"][1])): int(m["new_id"]) for m in data["merges"]
        }
        return tok
