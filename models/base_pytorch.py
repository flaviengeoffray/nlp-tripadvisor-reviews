from pathlib import Path
from typing import Any, List, Literal, Union, Dict

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .base import BaseModel

# @dataclass
# class BaseModelConfig:
#     type: str = "logistic-regression"
#     params: Dict[str, Any] = None


class BaseTorchModel(nn.Module, BaseModel):
    """
    Abstract base class for all models
    """

    def __init__(
        self,
        model_path: Path,
        **kwargs: Any,
        # device: str = "cpu",
        # model: nn.Module = None,
        # criterion: nn.Module = None,
        # optimizer: torch.optim.Optimizer = None,
        # scheduler: bool = True,
    ) -> None:

        nn.Module.__init__(self)
        BaseModel.__init__(self, model_path)

        lr: float = kwargs.pop("lr", 1e-3)

        self.device = torch.device(kwargs.pop("device", "cpu"))
        # self.model: nn.Module = kwargs.pop("model", None)
        self.criterion: nn.Module = kwargs.pop("criterion", nn.CrossEntropyLoss())
        self.optimizer: torch.optim.Optimizer = kwargs.pop(
            "optimizer", torch.optim.Adam(self.parameters(), lr=lr)
        )

        scheduler: bool = kwargs.pop("scheduler", True)

        self.scheduler: ReduceLROnPlateau = (
            ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2, verbose=True
            )
            if scheduler
            else None
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        patience: int = 5,
    ) -> None:

        self.to(self.device)
        patience_counter = 0
        best_val_loss = float("inf")

        for epoch in range(epochs):

            self.train()
            train_loss = 0.0

            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)

            self.eval()
            val_loss = 0.0
            all_metrics: List[Dict[str, float]] = []

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.forward(X)
                    loss = self.criterion(outputs, y)
                    metrics = self.evaluate(X, y, y_pred=outputs)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}"
            )

            metrics_means: Dict[str, float] = {}

            for k in all_metrics[0].keys():
                vals = [m[k] for m in all_metrics]
                metrics_means[k] = float(np.mean(vals))

            print(
                "Val Metrics — ",
                ", ".join(f"{k}={v:.4f}" for k, v in metrics_means.items()),
            )

            from utils import save_metrics  # FIXME: Crado

            save_metrics(
                metrics=metrics,
                epoch=epoch,
                path=self.model_path / "train_metrics.json",
            )

            if self.scheduler:
                self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save(self.model_path / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break

    # def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:

    #     self.model.eval()
    #     X = (
    #         torch.tensor(X, dtype=torch.float32)
    #         if not isinstance(X, Tensor)
    #         else X.float()
    #     ).to(self.device)

    #     with torch.no_grad():
    #         outputs = self.model(X)
    #         preds = torch.argmax(outputs, dim=1)
    #         return preds.cpu().numpy()

    # def evaluate(
    #     self,
    #     X: Union[np.ndarray, Tensor],
    #     y: Union[np.ndarray, Tensor],
    #     y_pred: Union[np.ndarray, Tensor] = None,
    # ) -> Dict[str, float]:
    #     # self.model.eval()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.to(self.device)
