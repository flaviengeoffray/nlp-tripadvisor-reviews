from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from data.tokenizers.base import TokenizerConfig
from models.base import BaseModelConfig
from vectorizers.base import VectorizerConfig


@dataclass
class Config:

    model_path: Path = Path("examples/LogisticRegression/")
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
    balance: bool = False
    balance_percentage: float = 0.8
    augmentation_methods: list[str] = None
