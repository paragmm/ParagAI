"""High-Speed Multilingual Indic + English BPE Tokenizer for ParagAI.

Extends BPETokenizer with:
- Configurable vocab sizes: 4096, 8192, 16384, 32768
- Explicit 256 UTF-8 byte fallback vocabulary
- ChatML special tokens (im_start, im_end, endoftext, tool_call)
- Memory-efficient chunked training for large multilingual corpora
- Per-script coverage statistics (Latin, Bengali, Devanagari, Tamil, Telugu)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator

from paragai.tokenizer.bpe_tokenizer import BPETokenizer

try:
    import regex as re

    _WORD_REGEX = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    import re  # type: ignore[no-redef]

    _WORD_REGEX = re.compile(r"""\s+|[^\s]+""")

# ---------------------------------------------------------------------------
# Unicode script ranges used for coverage stats
# ---------------------------------------------------------------------------
_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "Latin": (0x0000, 0x024F),
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
}

VALID_VOCAB_SIZES = (4096, 8192, 16384, 32768)

# ChatML & utility special tokens appended after the base-4 specials
CHATML_SPECIAL_TOKENS: list[str] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|tool_call|>",
]


def _detect_scripts(text: str) -> list[str]:
    """Return the list of script names whose characters appear in *text*."""
    found: list[str] = []
    for script, (lo, hi) in _SCRIPT_RANGES.items():
        if any(lo <= ord(ch) <= hi for ch in text):
            found.append(script)
    return found


def _chunk_text(text: str, chunk_size: int) -> Iterator[str]:
    """Yield non-overlapping UTF-8-safe chunks of roughly *chunk_size* bytes."""
    encoded = text.encode("utf-8")
    offset = 0
    total = len(encoded)
    while offset < total:
        end = min(offset + chunk_size, total)
        # Walk back to a valid UTF-8 boundary
        while end < total and (encoded[end] & 0xC0) == 0x80:
            end -= 1
        chunk_bytes = encoded[offset:end]
        yield chunk_bytes.decode("utf-8", errors="replace")
        offset = end


class IndicBPETokenizer(BPETokenizer):
    """Multilingual Indic + English BPE tokenizer with ChatML special tokens.

    Parameters
    ----------
    vocab_size:
        Target vocabulary size. Must be one of 4096, 8192, 16384, 32768.
    """

    # Additional special token IDs (follow the base 4 from BaseTokenizer)
    IM_START_ID: int = 4
    IM_END_ID: int = 5
    ENDOFTEXT_ID: int = 6
    TOOL_CALL_ID: int = 7

    def __init__(self, vocab_size: int = 4096) -> None:
        if vocab_size not in VALID_VOCAB_SIZES:
            raise ValueError(f"vocab_size must be one of {VALID_VOCAB_SIZES}, got {vocab_size}.")

        # Initialize parent (sets base 4 special tokens + 256 byte vocab)
        # but we override the special tokens list to include ChatML tokens.
        # We call super().__init__ with a dummy size to avoid doing extra work,
        # then rebuild the vocabulary ourselves.
        self.target_vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

        # --- 1. Base special tokens (IDs 0-3) ---
        self.special_tokens: list[str] = [
            self.PAD_TOKEN,  # 0
            self.UNK_TOKEN,  # 1
            self.BOS_TOKEN,  # 2
            self.EOS_TOKEN,  # 3
        ]

        # --- 2. ChatML special tokens (IDs 4-7) ---
        self.special_tokens.extend(CHATML_SPECIAL_TOKENS)

        # --- 3. Populate vocab: special tokens first, then all 256 bytes ---
        for idx, tok in enumerate(self.special_tokens):
            self.vocab[idx] = tok.encode("utf-8")

        # Byte fallback: IDs start immediately after special tokens
        self._byte_offset: int = len(self.special_tokens)
        for byte_val in range(256):
            self.vocab[self._byte_offset + byte_val] = bytes([byte_val])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size (special + bytes + learned merges)."""
        return len(self.vocab)

    @property
    def all_special_ids(self) -> list[int]:
        """Return all special token IDs."""
        return list(range(len(self.special_tokens)))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(  # type: ignore[override]
        self,
        texts: list[str] | str,
        verbose: bool = True,
        chunk_size: int = 10 * 1024 * 1024,  # 10 MB
    ) -> None:
        """Train BPE merge rules on multilingual text using word frequency hashing.

        Parameters
        ----------
        texts:
            A single string or a list of strings to train on.
        verbose:
            If True, print progress every 100 merges and final stats.
        chunk_size:
            Maximum bytes per processing chunk (default 10 MB).
        """
        if isinstance(texts, str):
            texts = [texts]

        num_merges = self.target_vocab_size - len(self.vocab)
        if num_merges <= 0:
            if verbose:
                print("Vocabulary already at target size; no merges needed.")
            return

        if verbose:
            total_chars = sum(len(t) for t in texts)
            detected = _detect_scripts("".join(texts[:1000]))
            print(
                f"[IndicBPE] Training on {total_chars:,} chars | "
                f"Scripts detected: {detected or ['Unknown']} | "
                f"Merges needed: {num_merges:,}"
            )

        # ---- Word-frequency aggregation ----
        word_freqs: Counter[str] = Counter()
        for text in texts:
            for chunk in _chunk_text(text, chunk_size):
                words = _WORD_REGEX.findall(chunk)
                word_freqs.update(words)

        if not word_freqs:
            if verbose:
                print("No words extracted from corpus; aborting training.")
            return

        word_list: list[list[int]] = []
        word_counts: list[int] = []
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)

        for w_idx, (word, count) in enumerate(word_freqs.items()):
            raw_bytes = list(word.encode("utf-8"))
            token_ids = [b + self._byte_offset for b in raw_bytes]
            word_list.append(token_ids)
            word_counts.append(count)
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                pair_counts[pair] += count
                pair_to_words[pair].add(w_idx)

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

                # Decrement old pairs
                for i in range(len(old_tokens) - 1):
                    p = (old_tokens[i], old_tokens[i + 1])
                    pair_counts[p] -= count
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    pair_to_words[p].discard(w_idx)
                    if not pair_to_words[p]:
                        del pair_to_words[p]

                # Merge occurrences in word
                new_tokens: list[int] = []
                i = 0
                while i < len(old_tokens):
                    if i < len(old_tokens) - 1 and old_tokens[i] == p0 and old_tokens[i + 1] == p1:
                        new_tokens.append(new_id)
                        i += 2
                    else:
                        new_tokens.append(old_tokens[i])
                        i += 1

                # Increment new pairs
                for i in range(len(new_tokens) - 1):
                    p = (new_tokens[i], new_tokens[i + 1])
                    pair_counts[p] += count
                    pair_to_words[p].add(w_idx)

                word_list[w_idx] = new_tokens

            if verbose and (m + 1) % 100 == 0:
                print(
                    f"  Merge {m + 1:>6}/{num_merges}: "
                    f"{best_pair} -> {new_id}  "
                    f"(vocab={len(self.vocab):,})"
                )

        if verbose:
            self._report_script_coverage(texts)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode *text* to a list of token IDs.

        Parameters
        ----------
        text:
            Input string (UTF-8 safe).
        add_special_tokens:
            If True, prepend BOS and append EOS token IDs.
        """
        if not text:
            return [self.BOS_ID, self.EOS_ID] if add_special_tokens else []

        ids = self._encode_chunk(text)
        if add_special_tokens:
            return [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        words = _WORD_REGEX.findall(chunk)
        ids: list[int] = []
        for word in words:
            raw_bytes = list(word.encode("utf-8"))
            word_ids = [b + self._byte_offset for b in raw_bytes]

            while len(word_ids) >= 2:
                stats = self.get_stats(word_ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                word_ids = self.merge(word_ids, pair, self.merges[pair])
            ids.extend(word_ids)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to a string.

        Parameters
        ----------
        ids:
            List of token IDs to decode.
        skip_special_tokens:
            If True, all special token IDs are dropped from output.
        """
        special_id_set = set(self.all_special_ids) if skip_special_tokens else set()
        byte_parts: list[bytes] = []
        for token_id in ids:
            if token_id in special_id_set:
                continue
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
            else:
                byte_parts.append(self.UNK_TOKEN.encode("utf-8"))
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Coverage stats
    # ------------------------------------------------------------------

    def get_coverage_stats(self, text: str) -> dict:
        """Return coverage statistics for *text*.

        Returns a dict with keys:
        - ``total_chars``: character count of the input.
        - ``total_tokens``: number of tokens the text encodes to.
        - ``unique_tokens``: count of distinct token IDs used.
        - ``avg_tokens_per_word``: mean tokens per whitespace-delimited word.
        - ``script``: detected Unicode scripts present in the text.
        - ``compression_ratio``: chars / tokens (higher = more compression).
        """
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
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, file_path: str) -> None:
        """Serialize tokenizer state to a JSON file.

        The ``"type"`` field is set to ``"bpe"`` for GGUF/export_gguf.py
        compatibility. The full type is recorded in ``"tokenizer_class"``.
        Merges are stored as ``[[p0, p1], new_id]`` pairs for compatibility
        with export_gguf.py's iteration pattern.
        """
        data: dict = {
            "type": "bpe",
            "tokenizer_class": "indic_bpe",
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self.special_tokens,
            "merges": [[int(p[0]), int(p[1]), int(v)] for p, v in self.merges.items()],
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str) -> IndicBPETokenizer:
        """Deserialize tokenizer state from a JSON file.

        Parameters
        ----------
        file_path:
            Path to a previously saved ``indic_bpe`` JSON file.
        """
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        tok = cls(vocab_size=data["target_vocab_size"])
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.special_tokens = data.get("special_tokens", tok.special_tokens)
        tok._byte_offset = len(tok.special_tokens)
        tok.merges = {(int(m[0]), int(m[1])): int(m[2]) for m in data["merges"]}
        return tok

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _report_script_coverage(self, texts: list[str]) -> None:
        """Print per-script coverage after training completes."""
        sample = "".join(texts)[:50_000]  # cap at 50 k chars for speed
        script_samples: dict[str, str] = {}
        for ch in sample:
            for script, (lo, hi) in _SCRIPT_RANGES.items():
                if lo <= ord(ch) <= hi:
                    script_samples.setdefault(script, "")
                    if len(script_samples[script]) < 500:
                        script_samples[script] += ch

        print("\n[IndicBPE] Script Coverage Report:")
        if not script_samples:
            print("  No recognised script ranges detected.")
            return
        for script, seg in script_samples.items():
            if not seg:
                continue
            stats = self.get_coverage_stats(seg)
            print(
                f"  {script:12s} | chars={stats['total_chars']:>5} "
                f"tokens={stats['total_tokens']:>5} "
                f"avg_tok/word={stats['avg_tokens_per_word']:>5} "
                f"ratio={stats['compression_ratio']:>5}"
            )
        print()
