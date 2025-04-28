from typing import Any, Optional, Sequence
from .base import BaseVectorizer


class Word2VecVectorizer(BaseVectorizer):

    def __init__(self, **kwargs: Any) -> None:
        self.vectorizer = None

    def fit(self, texts: Sequence[str], y: Optional[Any] = None) -> None:
        pass

    def transform(self, texts: Sequence[str]) -> Any:
        pass

    def inverse_transform(self, vectors: Any) -> Sequence[Sequence[str]]:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
