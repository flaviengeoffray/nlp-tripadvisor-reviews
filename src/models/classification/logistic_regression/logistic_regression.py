from pathlib import Path
from typing import Dict, Union, Any
import joblib
import numpy as np
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from models.classification.base import BaseClassificationModel
from my_tokenizers.base import BaseTokenizer
import logging

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

        logger.info("Metrics — %s", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        from utils import save_metrics

        save_metrics(
            metrics=metrics, epoch=None, path=self.model_path / "train_metrics.json"
        )

        self.save(self.model_path / "model.bz2")

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        X = X.detach().cpu().numpy() if isinstance(X, Tensor) else X
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
