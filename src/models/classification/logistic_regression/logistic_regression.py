import logging
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch import Tensor

from models.classification.base import BaseClassificationModel
from my_tokenizers.base import BaseTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseClassificationModel):
    def __init__(self, model_path: Path, **kwargs: Any) -> None:
        super().__init__(model_path)
        self.tokenizer: BaseTokenizer = kwargs.pop("tokenizer", None)
        self.model: LogisticRegression = LogisticRegression(**kwargs)

    def fit(
        self,
        X_train: Union[np.ndarray, Tensor],
        y_train: Union[np.ndarray, Tensor],
        X_val: Union[np.ndarray, Tensor],  # Validation features
        y_val: Union[np.ndarray, Tensor],  # Validation labels
    ) -> None:
        """
        Train the Logistic Regression model.

        :param X_train: Training features (document vectors)
        :param y_train: Training labels
        :param X_val: Validation features
        :param y_val: Validation labels
        """
        X_train = (
            X_train.detach().cpu().numpy() if isinstance(X_train, Tensor) else X_train
        )
        y_train = (
            y_train.detach().cpu().numpy() if isinstance(y_train, Tensor) else y_train
        )

        self.model.fit(X_train, y_train)
        metrics: Dict[str, float] = self.evaluate(X_train, y_train)

        logger.info(
            "Metrics — %s", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

        self.save(self.model_path / "model.bz2")

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Make predictions with the trained model.

        :param np.Union[np.ndarray, Tensor] X: Features to predict
        :return np.ndarray: Predicted class labels
        """
        X = X.detach().cpu().numpy() if isinstance(X, Tensor) else X
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        """
        Save the model to disk.

        :param Path path: Path to save the model
        """
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        """
        Load the model from disk.

        :param Path path: Path to load the model from
        """
        self.model = joblib.load(path)
