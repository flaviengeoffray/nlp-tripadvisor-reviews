import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from models.base import BaseModel, BaseModelConfig
from models.classification.bayes.naive_bayes import NaiveBayesModel
from models.classification.feedforward.feedforward import FNNModel
from models.classification.logistic_regression.logistic_regression import (
    LogisticRegressionModel,
)
from models.classification.lstm.lstm import LSTMModel
from models.classification.pretrained.pretrained import PretrainedClassifier
from models.classification.rnn.rnn import RNNClassifier
from models.generative.feedforward.feedforward import FNNGenerativeModel
from models.generative.ngram.ngram import NgramGenerator
from models.generative.pretrained.pretrained import PretrainedGenerator
from models.generative.rnn.rnn import RNNGenModel
from models.generative.transformer.transformer import Transformer
from my_tokenizers.base import BaseTokenizer, TokenizerConfig
from my_tokenizers.bpe import BpeTokenizer
from my_vectorizers.base import BaseVectorizer, VectorizerConfig
from my_vectorizers.tfidf import TfidfVectorizer
from my_vectorizers.word2vec import Word2VecVectorizer

from .config import Config

TOKENIZER_REGISTRY = {"bpe": BpeTokenizer}

VECTORIZER_REGISTRY = {"tf-idf": TfidfVectorizer, "word2vec": Word2VecVectorizer}

CLASSIFICATION_REGISTRY = {
    "naive-bayes": NaiveBayesModel,
    "logistic-regression": LogisticRegressionModel,
    "feedforward": FNNModel,
    "rnn-classification": RNNClassifier,
    "lstm": LSTMModel,
    "pre-trained": PretrainedClassifier,
}

GENERATIVE_REGISTRY = {
    "rnn_generator": RNNGenModel,
    "transformer": Transformer,
    "feedforward-generation": FNNGenerativeModel,
    "ngram": NgramGenerator,
    "pre-trained-gen": PretrainedGenerator,
}


def load_config(path: str) -> Config:
    """
    Load configuration from a YAML file.

    :param str path: Path to the YAML configuration file.
    :return Config: Config object populated with values from the file.
    """
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

    # Data prep
    test_size = data.get("test_size", Config.test_size)
    val_size = data.get("val_size", Config.val_size)
    seed = data.get("seed", Config.seed)
    stratify = data.get("stratify", Config.stratify)
    sample_size = data.get("sample_size", Config.sample_size)

    # Data augmentation
    balance = data.get("balance", Config.balance)
    balance_percentage = data.get("balance_percentage", Config.balance_percentage)
    augmentation_methods = data.get("augmentation_methods", Config.augmentation_methods)
    augmentation_workers = data.get("augmentation_workers", Config.augmentation_workers)
    augmented_data = data.get("augmented_data", Config.augmented_data)

    return Config(
        model_path=model_path,
        dataset_name=dataset_name,
        review_col=review_col,
        label_col=label_col,
        tokenizer=tokenizer_cfg,
        vectorizer=vectorizer_cfg,
        model=model_cfg,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        stratify=stratify,
        sample_size=sample_size,
        balance=balance,
        balance_percentage=balance_percentage,
        augmentation_methods=augmentation_methods,
        augmentation_workers=augmentation_workers,
        augmented_data=augmented_data,
    )


def load_tokenizer(cfg: TokenizerConfig) -> BaseTokenizer:
    """
    Load a tokenizer based on the provided configuration.

    :param TokenizerConfig cfg: Tokenizer configuration.
    :return BaseTokenizer: An instance of a tokenizer.
    """
    if cfg.type not in TOKENIZER_REGISTRY.keys():
        raise ValueError(f"Unknown tokenizer type: {cfg.type}")

    return TOKENIZER_REGISTRY[cfg.type](**(cfg.params or {}))


def load_vectorizer(cfg: VectorizerConfig) -> BaseVectorizer:
    """
    Load a vectorizer based on the provided configuration.

    :param VectorizerConfig cfg: Vectorizer configuration.
    :return BaseVectorizer: An instance of a vectorizer.
    """
    if cfg.type not in VECTORIZER_REGISTRY.keys():
        raise ValueError(f"Unknown vectorizer type: {cfg.type}")

    return VECTORIZER_REGISTRY[cfg.type](**(cfg.params or {}))


def load_model(cfg: BaseModelConfig, model_path: Path) -> BaseModel:
    """
    Load a model based on the provided configuration.

    :param BaseModelConfig cfg: Model configuration.
    :param Path model_path: Path to the model file.
    :return BaseModel: An instance of a model.
    """
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
    """
    Save metrics to a JSON file.

    :param Dict[str, Any] metrics: Metrics to save.
    :param int epoch: Current epoch number (optional).
    :param str path: Path to the JSON file (default is "metrics.json").
    """
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
