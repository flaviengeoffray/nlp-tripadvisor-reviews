from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import numpy as np
from torch import Tensor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
