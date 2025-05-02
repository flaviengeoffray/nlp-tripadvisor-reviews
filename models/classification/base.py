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

        if not y_pred:
            y_pred = self.predict(X)
        y_true = y.detach().cpu().numpy() if isinstance(y, Tensor) else np.asarray(y)
        y_pred_np = (
            y_pred.detach().cpu().numpy()
            if isinstance(y_pred, Tensor)
            else np.asarray(y_pred)
        )

        return {
            "accuracy": accuracy_score(y_true, y_pred_np),
            "precision": precision_score(
                y_true, y_pred_np, average="macro", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred_np, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred_np, average="macro", zero_division=0),
        }
