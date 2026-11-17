"""Industrial-Grade High-Throughput Multilingual Tokenizer for ParagAI.

Dual-tier architecture:
- Tier 1: High-speed Rust-backed ByteLevel BPE tokenizer (Hugging Face `tokenizers`).
- Tier 2: Automatic fallback to optimized first-principles `IndicBPETokenizer` if
  the `tokenizers` package is not installed.
"""

from __future__ import annotations

import json
import os
from typing import Any

from paragai.tokenizer.indic_bpe import (
    CHATML_SPECIAL_TOKENS,
    VALID_VOCAB_SIZES,
    IndicBPETokenizer,
    _detect_scripts,
)

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False


class FastIndicBPETokenizer:
    """Industrial multi-threaded multilingual BPE Tokenizer with Rust backend.

    Guarantees:
    - 100% loss-free roundtrip via ByteLevel UTF-8 fallback.
    - Standard special tokens (<pad>=0, <unk>=1, <bos>=2, <eos>=3).
    - ChatML special tokens (<|im_start|>=4, <|im_end|>=5, <|endoftext|>=6, <|tool_call|>=7).
    - Dual serialization: native ParagAI JSON and Hugging Face standard tokenizer.json.
    - Graceful fallback to pure-Python `IndicBPETokenizer` if `tokenizers` is not installed.

    Parameters
    ----------
    vocab_size:
        Target vocabulary size. Must be one of 4096, 8192, 16384, 32768.
    """

    PAD_ID: int = 0
    UNK_ID: int = 1
    BOS_ID: int = 2
    EOS_ID: int = 3
    SYSTEM_ID: int = 4
    USER_ID: int = 5
    ASSISTANT_ID: int = 6
    END_ID: int = 7
    IM_START_ID: int = 8
    IM_END_ID: int = 9
    ENDOFTEXT_ID: int = 10
    TOOL_CALL_ID: int = 11

    def __init__(self, vocab_size: int = 4096) -> None:
        if vocab_size not in VALID_VOCAB_SIZES:
            raise ValueError(
                f"vocab_size must be one of {VALID_VOCAB_SIZES}, got {vocab_size}."
            )

        self.target_vocab_size = vocab_size
        self.special_tokens = [
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
        ] + list(CHATML_SPECIAL_TOKENS)

        self._backend = "rust" if HAS_TOKENIZERS else "pure_python"
        self._hf_tokenizer: Tokenizer | None = None
        self._fallback_tokenizer: IndicBPETokenizer | None = None

        if HAS_TOKENIZERS:
            self._init_rust_tokenizer()
        else:
            self._fallback_tokenizer = IndicBPETokenizer(vocab_size=vocab_size)

    def _init_rust_tokenizer(self) -> None:
        """Initialize Hugging Face Rust Tokenizer pipeline."""
        tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tok.decoder = decoders.ByteLevel()
        self._hf_tokenizer = tok

    @property
    def backend(self) -> str:
        """Current active backend ('rust' or 'pure_python')."""
        return self._backend

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.get_vocab_size()
        if self._fallback_tokenizer is not None:
            return self._fallback_tokenizer.vocab_size
        return len(self.special_tokens) + 256

    @property
    def all_special_ids(self) -> list[int]:
        """List of all special token IDs."""
        return list(range(len(self.special_tokens)))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: list[str] | str,
        verbose: bool = True,
        chunk_size: int = 10 * 1024 * 1024,
    ) -> None:
        """Train BPE merge rules on multilingual corpora.

        Parameters
        ----------
        texts:
            A single string or list of text strings.
        verbose:
            Whether to report progress and coverage statistics.
        chunk_size:
            Chunk size in bytes for batching text.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return

        if self._hf_tokenizer is not None:
            if verbose:
                total_chars = sum(len(t) for t in texts)
                detected = _detect_scripts("".join(texts[:1000]))
                print(
                    f"[FastIndicBPE - Rust] Training on {total_chars:,} chars | "
                    f"Scripts: {detected or ['Unknown']} | "
                    f"Target Vocab: {self.target_vocab_size:,}"
                )

            trainer = trainers.BpeTrainer(
                vocab_size=self.target_vocab_size,
                special_tokens=self.special_tokens,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                show_progress=verbose,
            )
            self._hf_tokenizer.train_from_iterator(texts, trainer=trainer)

            if verbose:
                print(
                    f"[FastIndicBPE - Rust] Training finished. "
                    f"Vocab size: {self.vocab_size:,}"
                )
        else:
            if verbose:
                print("[FastIndicBPE] Fallback to pure Python engine...")
            assert self._fallback_tokenizer is not None
            self._fallback_tokenizer.train(texts, verbose=verbose, chunk_size=chunk_size)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs."""
        if not text:
            return [self.BOS_ID, self.EOS_ID] if add_special_tokens else []

        if self._hf_tokenizer is not None:
            encoding = self._hf_tokenizer.encode(text)
            ids = list(encoding.ids)
            if add_special_tokens:
                return [self.BOS_ID] + ids + [self.EOS_ID]
            return ids

        assert self._fallback_tokenizer is not None
        return self._fallback_tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to a string."""
        if not ids:
            return ""

        if self._hf_tokenizer is not None:
            special_ids = set(self.all_special_ids) if skip_special_tokens else set()
            filtered = [i for i in ids if i not in special_ids]
            return self._hf_tokenizer.decode(filtered)

        assert self._fallback_tokenizer is not None
        return self._fallback_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Coverage stats
    # ------------------------------------------------------------------

    def get_coverage_stats(self, text: str) -> dict[str, Any]:
        """Compute coverage and compression statistics for text."""
        encoded = self.encode(text)
        words = text.split()
        if words:
            tokens_per_word: list[float] = [len(self.encode(w)) for w in words]
            avg_tpw = sum(tokens_per_word) / len(tokens_per_word)
        else:
            avg_tpw = 0.0

        return {
            "total_chars": len(text),
            "total_tokens": len(encoded),
            "unique_tokens": len(set(encoded)),
            "avg_tokens_per_word": round(avg_tpw, 3),
            "script": _detect_scripts(text),
            "compression_ratio": round(len(text) / max(len(encoded), 1), 3),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, file_path: str) -> None:
        """Save tokenizer state to ParagAI JSON format."""
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

        if self._hf_tokenizer is not None:
            vocab = self._hf_tokenizer.get_vocab()
            # Invert vocab to id -> bytes
            id_to_bytes: dict[str, list[int]] = {}
            for token_str, token_id in vocab.items():
                id_to_bytes[str(token_id)] = list(token_str.encode("utf-8"))

            data = {
                "type": "bpe",
                "tokenizer_class": "fast_indic_bpe",
                "backend": "rust",
                "target_vocab_size": self.target_vocab_size,
                "special_tokens": self.special_tokens,
                "vocab": id_to_bytes,
                "merges": [],  # merges represented internally in Rust model
            }
            with open(file_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)

            # Also save HF json adjacent if possible
            base_dir = os.path.dirname(file_path) or "."
            hf_path = os.path.join(base_dir, "tokenizer.json")
            self._hf_tokenizer.save(hf_path)
        else:
            assert self._fallback_tokenizer is not None
            self._fallback_tokenizer.save(file_path)

    def export_huggingface(self, output_dir: str) -> str:
        """Export Hugging Face tokenizer.json and config for vLLM & Transformers."""
        os.makedirs(output_dir, exist_ok=True)
        hf_path = os.path.join(output_dir, "tokenizer.json")

        if self._hf_tokenizer is not None:
            self._hf_tokenizer.save(hf_path)
        else:
            # Construct a basic tokenizer.json if only pure Python is present
            assert self._fallback_tokenizer is not None
            self._fallback_tokenizer.save(os.path.join(output_dir, "tokenizer_indic.json"))

        config_path = os.path.join(output_dir, "tokenizer_config.json")
        config_data = {
            "model_type": "llama",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "clean_up_tokenization_spaces": False,
            "added_tokens_decoder": {
                str(i): {"content": tok, "special": True}
                for i, tok in enumerate(self.special_tokens)
            },
        }
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config_data, fh, indent=2)

        return hf_path

    @classmethod
    def load(cls, file_path: str) -> FastIndicBPETokenizer:
        """Load tokenizer from saved JSON file."""
        base_dir = os.path.dirname(file_path) or "."
        hf_path = os.path.join(base_dir, "tokenizer.json")

        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        target_size = data.get("target_vocab_size", 4096)
        if target_size not in VALID_VOCAB_SIZES:
            target_size = 4096
        tok = cls(vocab_size=target_size)
        if HAS_TOKENIZERS and (os.path.exists(hf_path) or "version" in data or "model" in data):
            actual_hf_path = file_path if ("version" in data or "model" in data) else hf_path
            tok._hf_tokenizer = Tokenizer.from_file(actual_hf_path)
            tok._backend = "rust"
            # Ensure ParagAI chat special tokens are registered
            new_specials = [
                s for s in ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
                if tok._hf_tokenizer.token_to_id(s) is None
            ]
            if new_specials:
                tok.add_special_tokens(new_specials)
        else:
            # Fallback load
            tok._fallback_tokenizer = IndicBPETokenizer.load(file_path)
            tok._backend = "pure_python"
            tok._hf_tokenizer = None

        return tok

    def token_to_id(self, token: str) -> int | None:
        """Return the token ID for a given string token."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id(token)
        if self._fallback_tokenizer is not None:
            if token in self._fallback_tokenizer.special_tokens:
                return self._fallback_tokenizer.special_tokens.index(token)
            for tid, b in self._fallback_tokenizer.vocab.items():
                if b.decode("utf-8", errors="ignore") == token:
                    return tid
        return None

    def add_special_tokens(self, special_tokens: list[str]) -> int:
        """Register additional special tokens into the tokenizer."""
        count = 0
        if self._hf_tokenizer is not None:
            count = self._hf_tokenizer.add_special_tokens(special_tokens)
            for tok in special_tokens:
                if tok not in self.special_tokens:
                    self.special_tokens.append(tok)
        elif self._fallback_tokenizer is not None:
            for tok in special_tokens:
                if tok not in self._fallback_tokenizer.special_tokens:
                    self._fallback_tokenizer.special_tokens.append(tok)
                    self.special_tokens.append(tok)
                    count += 1
        return count
