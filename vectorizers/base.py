from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


@dataclass
class VectorizerConfig:
    type: str = "tf-idf"
    checkpoint: Path = None
    params: Dict[str, Any] = None


class BaseVectorizer(ABC):
    @abstractmethod
    def fit(self, texts: Sequence[str], y: Optional[Any] = None) -> None:
        pass

    def fit_transform(self, texts: Sequence[str], y: Optional[Any] = None) -> Any:
        self.fit(texts, y)
        return self.transform(texts)

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> Any:
        pass

    @abstractmethod
    def inverse_transform(self, vectors: Any) -> Sequence[str]:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        pass
