from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.base import BaseModelConfig
from my_tokenizers.base import TokenizerConfig
from my_vectorizers.base import VectorizerConfig


@dataclass
class Config:
    model_path: Path = Path("configurations/LogisticRegression/")
    dataset_name: str = "jniimi/tripadvisor-review-rating"
    review_col: str = "review"
    label_col: str = "overall"
    tokenizer: Optional[TokenizerConfig] = None
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    model: BaseModelConfig = field(default_factory=BaseModelConfig)

    # Data preparation
    test_size: float = 0.2
    val_size: float = 0.2
    seed: int = 42
    stratify: bool = True
    sample_size: Optional[int] = None

    # Data augmentation
    augmented_data: Optional[str] = None
    balance: bool = False
    balance_percentage: float = 0.8
    augmentation_methods: list[str] = None
    augmentation_workers: Optional[int] = None

    def get(self, key: str, default=None):
        if not hasattr(self, key) and default is None:
            raise KeyError(f"Key '{key}' missing in Config.")
        return getattr(self, key, default)
