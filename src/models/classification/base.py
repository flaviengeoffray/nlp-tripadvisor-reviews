from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import Tensor

from models.base import BaseModel


class BaseClassificationModel(BaseModel, ABC):
    """
    Base class for classification models.
    Model can be torch or numpy model.
    """

    @abstractmethod
    def predict(self, X: Union[np.ndarray, Tensor]) -> Any:
        raise NotImplementedError

    def evaluate(
        self,
        X: Union[np.ndarray, Tensor],
        y: Union[np.ndarray, Tensor],
        y_pred: Union[np.ndarray, Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model's performance.

        :param Union[np.ndarray, Tensor] X: Features to evaluate
        :param Union[np.ndarray, Tensor] y: True labels
        :param Union[np.ndarray, Tensor] y_pred: Predicted labels (if already computed
            to avoid recomputation)
        :return Dict[str, float]: Dictionary with accuracy, precision, recall, and f1
        """
        if y_pred is None:
            y_pred = self.predict(X)

        if isinstance(y_pred, Tensor):
            y_pred_np = y_pred.detach().cpu().numpy()
        else:
            y_pred_np = np.asarray(y_pred)

        if y_pred_np.ndim == 2:
            y_pred_np = np.argmax(y_pred_np, axis=1)

        if isinstance(y, Tensor):
            y_true = y.detach().cpu().numpy()
        else:
            y_true = np.asarray(y)

        return {
            "accuracy": accuracy_score(y_true, y_pred_np),
            "precision": precision_score(
                y_true, y_pred_np, average="macro", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred_np, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred_np, average="macro", zero_division=0),
        }
