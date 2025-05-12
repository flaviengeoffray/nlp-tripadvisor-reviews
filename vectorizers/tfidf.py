from pathlib import Path
import joblib
from typing import Any, Optional, Sequence
from data.tokenizers.base import BaseTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from .base import BaseVectorizer


class TfidfVectorizer(BaseVectorizer):

    def __init__(self, **kwargs: Any) -> None:
        self.tokenizer: BaseTokenizer = kwargs.pop("tokenizer", None)
        self.vectorizer: SklearnTfidfVectorizer = (
            SklearnTfidfVectorizer(**kwargs, tokenizer=self.tokenizer.tokenize)
            if self.tokenizer is not None
            else SklearnTfidfVectorizer(**kwargs)
        )

    def fit(self, texts: Sequence[str], y: Optional[Any] = None) -> None:
        self.vectorizer.fit(texts)

    def transform(self, texts: Sequence[str]) -> Any:
        return self.vectorizer.transform(texts)

    def inverse_transform(self, vectors: Any) -> Sequence[Sequence[str]]:
        return self.vectorizer.inverse_transform(vectors)

    def save(self, path: Path) -> None:
        joblib.dump(self.vectorizer, path)

    def load(self, path: Path) -> None:
        self.vectorizer = joblib.load(path)
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.vectorizer.tokenizer = self.tokenizer.tokenize
