from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Union, Dict

import numpy as np
from torch import Tensor


@dataclass
class BaseModelConfig:
    type: str = "logistic-regression"
    checkpoint: Path = None
    params: Dict[str, Any] = None


class BaseModel(ABC):
    """
    Abstract base class for all models
    """

    def __init__(self, model_path: Path, **kwargs: Any) -> None:
        self.model_path: Path = model_path

    @abstractmethod
    def fit(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        X: Union[np.ndarray, Tensor],
        y: Union[np.ndarray, Tensor],
        y_pred: Union[np.ndarray, Tensor] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError


# class BaseModel(ABC):
#     """
#     Abstract base class for all models (PyTorch and NumPy).
#     Defines a unified interface: train, predict, save, load.
#     """

#     def __init__(self, config):
#         self.config = config

#     @abstractmethod
#     def train(self, *args, **kwargs):
#         pass

#     @abstractmethod
#     def predict(self, X):
#         pass

#     @abstractmethod
#     def save(self, path: str = None):
#         pass

#     @abstractmethod
#     def load(self, path: str):
#         pass
