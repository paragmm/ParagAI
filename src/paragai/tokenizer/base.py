"""Abstract Base Class for ParagAI Tokenizers."""

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract interface that all ParagAI tokenizers must implement."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    SYSTEM_TOKEN = "<|system|>"
    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    END_TOKEN = "<|end|>"

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    SYSTEM_ID = 4
    USER_ID = 5
    ASSISTANT_ID = 6
    END_ID = 7

    SPECIAL_TOKENS: list[str] = [
        PAD_TOKEN,
        UNK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        SYSTEM_TOKEN,
        USER_TOKEN,
        ASSISTANT_TOKEN,
        END_TOKEN,
    ]

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
