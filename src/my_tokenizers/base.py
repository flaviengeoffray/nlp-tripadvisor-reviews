from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TokenizerConfig:
    type: str = "bpe"
    checkpoint: Path = None
    params: Dict[str, Any] = None


class BaseTokenizer(ABC):
    """
    Base class for tokenziers
    """

    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    @abstractmethod
    def token_to_id(self, token: str) -> Optional[int]:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        pass
