"""Abstract Base Class for ParagAI Tokenizers."""

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract interface that all ParagAI tokenizers must implement."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Convert string to list of token IDs."""

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Convert list of token IDs back into string."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return total vocabulary size."""

    @abstractmethod
    def save(self, file_path: str) -> None:
        """Serialize tokenizer state to disk."""

    @classmethod
    @abstractmethod
    def load(cls, file_path: str) -> "BaseTokenizer":
        """Deserialize tokenizer state from disk."""
