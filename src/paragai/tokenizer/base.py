"""Abstract Base Class for ParagAI Tokenizers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


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
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convert string to list of token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Convert list of token IDs back into string."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """Serialize tokenizer state to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, file_path: str) -> "BaseTokenizer":
        """Deserialize tokenizer state from disk."""
        pass

