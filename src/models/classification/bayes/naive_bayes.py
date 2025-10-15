import numpy as np
import pickle
from pathlib import Path
from typing import Any, Union

from sklearn.naive_bayes import MultinomialNB

from models.classification.base import BaseClassificationModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class NaiveBayesModel(BaseClassificationModel):
    """
    Naive Bayes classifier for text classification.
    Uses scikit-learn's MultinomialNB implementation.
    """

    def __init__(
        self,
        model_path: Path,
        alpha: float = 1.0,
        fit_prior: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Naive Bayes model.

        Args:
            model_path: Path to save/load model
            alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)
            fit_prior: Whether to learn class prior probabilities or not
            **kwargs: Additional keyword arguments
        """
        super().__init__(model_path, **kwargs)

        self.model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

        # Create directories if needed
        self.model_path.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any = None,
        y_val: Any = None,
    ) -> None:
        """
        Train the Naive Bayes model.

        Args:
            X_train: Training features (document vectors)
            y_train: Training labels
            X_val: Validation features (not used in Naive Bayes but kept for interface consistency)
            y_val: Validation labels (not used in Naive Bayes but kept for interface consistency)
        """
        # Convert to numpy arrays if not already
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()

        if isinstance(y_train, list) and all(
            isinstance(label, str) for label in y_train[:5]
        ):
            # For string labels, convert to integers (1-5)
            # Assuming ratings from 1-5
            label_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
            y_train = np.array([label_map.get(str(y), int(y) - 1) for y in y_train])
        else:
            # For numeric labels, adjust to 0-4 range
            y_train = np.array(y_train, dtype=int) - 1

        # Train the model
        logger.info("Training Naive Bayes model...")
        self.model.fit(X_train, y_train)

        # If validation data is provided, evaluate and print metrics
        if X_val is not None and y_val is not None:
            if hasattr(X_val, "toarray"):
                X_val = X_val.toarray()

            if isinstance(y_val, list) and all(
                isinstance(label, str) for label in y_val[:5]
            ):
                y_val = np.array([label_map.get(str(y), int(y) - 1) for y in y_val])
            else:
                y_val = np.array(y_val, dtype=int) - 1

            y_pred = self.predict(X_val)
            metrics = self.evaluate(X_val, y_val, y_pred)

            logger.info("Validation Metrics:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")

            # Save metrics
            from utils import save_metrics

            save_metrics(
                metrics=metrics,
                epoch=0,  # Naive Bayes has no epochs
                path=self.model_path / "train_metrics.json",
            )

        # Save the trained model
        self.save(self.model_path / "model.pkl")
        logger.info(f"Model saved to {self.model_path / 'model.pkl'}")

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            X: Features to predict

        Returns:
            Predicted class probabilities
        """
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Get probability distributions
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path) -> None:
        """
        Load the model from disk.

        Args:
            path: Path to load the model from
        """
        with open(path, "rb") as f:
            self.model = pickle.load(f)
