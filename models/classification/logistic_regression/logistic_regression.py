from pathlib import Path
from typing import Dict, Union, Any
import joblib
import numpy as np
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from models.classification.base import BaseClassificationModel


class LogisticRegressionModel(BaseClassificationModel):

    def __init__(self, model_path: Path, **kwargs: Any):
        super().__init__(model_path)
        self.model = LogisticRegression(**kwargs)

    def fit(
        self,
        X_train: Union[np.ndarray, Tensor],
        y_train: Union[np.ndarray, Tensor],
        X_val: Union[np.ndarray, Tensor],
        y_val: Union[np.ndarray, Tensor],
    ) -> None:
        X_train = (
            X_train.detach().cpu().numpy() if isinstance(X_train, Tensor) else X_train
        )
        y_train = (
            y_train.detach().cpu().numpy() if isinstance(y_train, Tensor) else y_train
        )
        self.model.fit(X_train, y_train)
        metrics: Dict[str, float] = self.evaluate(X_train, y_train)

        print("Metrics —", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        from utils import save_metrics  # FIXME: Crado

        save_metrics(
            metrics=metrics, epoch=None, path=self.model_path / "train_metrics.json"
        )

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        X = X.detach().cpu().numpy() if isinstance(X, Tensor) else X
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)

    # def evaluate(\
    #     self, X: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]
    # ) -> float:
    #     y_pred = self.predict(X)
    #     y_true = y.detach().cpu().numpy() if isinstance(y, Tensor) else y
    #     return accuracy_score(y_true, y_pred)
