import json
import os
from pathlib import Path
from typing import Any, Dict
import yaml
from data.tokenizers.base import TokenizerConfig
from models.generative.ngram.ngram import NgramGenerator
from vectorizers.base import VectorizerConfig
from models.base import BaseModelConfig, BaseModel
from config import Config

from data.tokenizers.base import BaseTokenizer
from data.tokenizers.bpe import BpeTokenizer
from vectorizers.base import BaseVectorizer
from vectorizers.tfidf import TfidfVectorizer

from vectorizers.word2vec import Word2VecVectorizer

from models.classification.logistic_regression.logistic_regression import (
    LogisticRegressionModel,
)
from models.classification.feedforward.feedforward import FNNModel
from models.classification.rnn.rnn import RNNModel
from models.classification.lstm.lstm import LSTMModel

from models.generative.transformer.transformer import Transformer


TOKENIZER_REGISTRY = {"bpe": BpeTokenizer}

VECTORIZER_REGISTRY = {"tf-idf": TfidfVectorizer, "word2vec": Word2VecVectorizer}

CLASSIFICATION_REGISTRY = {
    "logistic-regression": LogisticRegressionModel,
    "feedforward": FNNModel,
    "rnn": RNNModel,
    "lstm": LSTMModel,
}

GENERATIVE_REGISTRY = {"transformer": Transformer, "ngram": NgramGenerator}


def load_config(path: str) -> Config:

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    model_path = Path(data.get("model_path", Config.model_path))
    dataset_name = data.get("dataset_name", Config.dataset_name)
    review_col = data.get("review_col", Config.review_col)
    label_col = data.get("label_col", Config.label_col)

    tokenizer_cfg = (
        TokenizerConfig(**data.get("tokenizer", {})) if data.get("tokenizer") else None
    )
    vectorizer_cfg = (
        VectorizerConfig(**data.get("vectorizer", {}))
        if data.get("vectorizer")
        else None
    )
    model_cfg = BaseModelConfig(**data.get("model", {}))

    return Config(
        model_path=model_path,
        dataset_name=dataset_name,
        review_col=review_col,
        label_col=label_col,
        tokenizer=tokenizer_cfg,
        vectorizer=vectorizer_cfg,
        model=model_cfg,
    )


def load_tokenizer(cfg: TokenizerConfig) -> BaseTokenizer:
    if cfg.type not in TOKENIZER_REGISTRY.keys():
        raise ValueError(f"Unknown tokenizer type: {cfg.type}")

    return TOKENIZER_REGISTRY[cfg.type](**(cfg.params or {}))


def load_vectorizer(cfg: VectorizerConfig) -> BaseVectorizer:
    if cfg.type not in VECTORIZER_REGISTRY.keys():
        raise ValueError(f"Unknown vectorizer type: {cfg.type}")

    return VECTORIZER_REGISTRY[cfg.type](**(cfg.params or {}))


def load_model(cfg: BaseModelConfig, model_path: Path) -> BaseModel:
    model_type = cfg.type.lower()

    ModelClass: BaseModel

    if model_type in CLASSIFICATION_REGISTRY:
        ModelClass = CLASSIFICATION_REGISTRY[model_type]
    elif model_type in GENERATIVE_REGISTRY:
        ModelClass = GENERATIVE_REGISTRY[model_type]
    else:
        raise ValueError(f"Unknown model type: {cfg.type!r}")

    return ModelClass(model_path, **(cfg.params or {}))


def save_metrics(
    metrics: Dict[str, Any], epoch: int = None, path: str = "metrics.json"
) -> None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    if epoch:
        entry = {"epoch": epoch, **metrics}
        history.append(entry)
    else:
        history.append(metrics)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
