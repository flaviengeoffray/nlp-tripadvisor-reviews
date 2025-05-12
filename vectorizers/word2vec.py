from pathlib import Path
from typing import Any, Optional, Sequence

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

from vectorizers.base import BaseVectorizer
from data.tokenizers.base import BaseTokenizer


class Word2VecVectorizer(BaseVectorizer):
    def __init__(self, **kwargs):
        self.vector_size = kwargs.pop("vector_size", 100)
        self.window = kwargs.pop("window", 5)
        self.min_count = kwargs.pop("min_count", 1)
        self.workers = kwargs.pop("workers", 4)
        self.tokenizer: BaseTokenizer = kwargs.pop("tokenizer", None)
        self.max_len = kwargs.pop("max_len", 128)
        self.model: Optional[Word2Vec] = None

    def fit(self, texts: Sequence[str], y: Optional[Any] = None) -> None:
        if self.tokenizer:
            sentences = [self.tokenizer.tokenize(text) for text in texts]
        else:
            sentences = [simple_preprocess(text) for text in texts]
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )

    def transform(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self.model is None:
            raise ValueError("Word2Vec model has not been fitted yet.")
        if self.tokenizer:
            tokens = [self.tokenizer.tokenize(text) for text in texts]
        else:
            tokens = [simple_preprocess(text) for text in texts]
        # return [self._average_vector(tokens) for tokens in sentences]
        all_vecs = []
        for tokens in tokens:
            vecs = [
                (
                    self.model.wv[word]
                    if word in self.model.wv
                    else np.zeros(self.vector_size, dtype=float)
                )
                for word in tokens
            ]
            if self.max_len is not None:
                if len(vecs) > self.max_len:
                    vecs = vecs[: self.max_len]
                elif len(vecs) < self.max_len:
                    padding = [np.zeros(self.vector_size, dtype=float)] * (
                        self.max_len - len(vecs)
                    )
                    vecs = vecs + padding
            all_vecs.append(
                np.vstack(vecs)
                if vecs
                else np.zeros((self.max_len or 1, self.vector_size))
            )
        return all_vecs

    def _average_vector(self, tokens: Sequence[str]) -> Sequence[float]:
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        if not vectors:
            return [0.0] * self.vector_size
        mean = sum(vectors) / len(vectors)
        return mean.tolist() if hasattr(mean, "tolist") else list(mean)

    def inverse_transform(
        self, vectors: Sequence[Sequence[float]]
    ) -> Sequence[Sequence[str]]:
        if self.model is None:
            raise ValueError("Word2Vec model has not been fitted yet.")
        topn = 10
        results = []
        for vec in vectors:
            similar = self.model.wv.similar_by_vector(vec, topn=topn)
            words = [word for word, _ in similar]
            if self.tokenizer is not None:
                ids = [self.tokenizer.token_to_id(tok) for tok in words]
                text = self.tokenizer.decode(ids)
                results.append(text)
            else:
                results.append(words)

        return results

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("No model available to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: Path) -> None:
        path = Path(path)
        self.model = Word2Vec.load(str(path))
        self.vector_size = self.model.vector_size
