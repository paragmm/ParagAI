"""Character-level tokenizer for ParagAI."""

import json

from paragai.tokenizer.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """Simple, deterministic character-level tokenizer."""

    def __init__(self, chars: list[str] | None = None):
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

        # Initialize with special tokens
        self.stoi: dict[str, int] = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.itos: dict[int, str] = {idx: tok for idx, tok in enumerate(self.special_tokens)}

        if chars:
            self.build_vocab(chars)

    def build_vocab(self, chars: list[str]) -> None:
        """Add unique characters to vocabulary."""
        unique_chars = sorted(set(chars))
        for ch in unique_chars:
            if ch not in self.stoi:
                idx = len(self.stoi)
                self.stoi[ch] = idx
                self.itos[idx] = ch

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.BOS_ID)

        for ch in text:
            tokens.append(self.stoi.get(ch, self.UNK_ID))

        if add_special_tokens:
            tokens.append(self.EOS_ID)

        return tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        res = []
        for i in ids:
            if skip_special_tokens and i in (self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID):
                continue
            res.append(self.itos.get(i, self.UNK_TOKEN))
        return "".join(res)

    def save(self, file_path: str) -> None:
        data = {
            "type": "char",
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "CharTokenizer":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        return tok
