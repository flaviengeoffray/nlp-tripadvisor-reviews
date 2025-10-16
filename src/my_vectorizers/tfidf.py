from pathlib import Path
from typing import Any, Sequence

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

from my_tokenizers.base import BaseTokenizer

from .base import BaseVectorizer


class TfidfVectorizer(BaseVectorizer):
    def __init__(self, **kwargs: Any) -> None:
        self.tokenizer: BaseTokenizer = kwargs.pop("tokenizer", None)
        self.vectorizer: SklearnTfidfVectorizer = (
            SklearnTfidfVectorizer(**kwargs, tokenizer=self.tokenizer.tokenize)
            if self.tokenizer is not None
            else SklearnTfidfVectorizer(**kwargs)
        )

    def fit(self, texts: Sequence[str]) -> None:
        """
        Fit the TF-IDF vectorizer on a list of texts.

        :param Sequence[str] texts: List of input texts to train the vectorizer
        """
        self.vectorizer.fit(texts)

    def transform(self, texts: Sequence[str]) -> Any:
        """
        Transform a list of texts into TF-IDF feature vectors.

        :param Sequence[str] texts: List of input texts to transform
        :return Any: Transformed TF-IDF feature vectors
        """
        return self.vectorizer.transform(texts)

    def inverse_transform(self, vectors: Any) -> Sequence[Sequence[str]]:
        """
        Inverse transform TF-IDF feature vectors back to original texts.

        :param Any vectors: TF-IDF feature vectors to inverse transform
        :return Sequence[Sequence[str]]: List of original texts
        """
        return self.vectorizer.inverse_transform(vectors)

    def save(self, path: Path) -> None:
        """
        Save the vectorizer to a file.

        :param Path path: File path to save the vectorizer
        """
        joblib.dump(self.vectorizer, path)

    def load(self, path: Path) -> None:
        """
        Load the vectorizer from a file.

        :param Path path: File path to load the vectorizer from
        """
        self.vectorizer = joblib.load(path)
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.vectorizer.tokenizer = self.tokenizer.tokenize
