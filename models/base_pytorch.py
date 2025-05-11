from pathlib import Path
from abc import abstractmethod
from typing import Any, List, Tuple, Dict

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .base import BaseModel
from data.tokenizers.base import BaseTokenizer

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

        # nn.Module.__init__(self)
        BaseModel.__init__(self, model_path)

        lr: float = kwargs.pop("lr", 1e-4)

        device_str = kwargs.pop("device", "cpu")
        if device_str == "mps" and not torch.backends.mps.is_available():
            print(
                "Warning: MPS (Apple GPU) requested but not available. Falling back to CPU."
            )
            device_str = "cpu"
        self.device = torch.device(device_str)
        print(f"Using device: {self.device.type}")
        if self.device.type == "cuda":
            print(f"    Device name: {torch.cuda.get_device_name(self.device)}")
        elif self.device.type == "mps":
            print("    Using Apple Silicon GPU (MPS backend)")
        self.to(self.device)

        self.tokenizer: BaseTokenizer = kwargs.pop("tokenizer", None)

        self.start_epoch: int = 0
        self.epochs: int = kwargs.pop("epochs", 20)
        self.patience: int = kwargs.pop("patience", 5)
        # self.model: nn.Module = kwargs.pop("model", None)
        self.criterion: nn.Module = kwargs.pop("criterion", nn.CrossEntropyLoss()).to(
            self.device
        )
        self.optimizer: torch.optim.Optimizer = kwargs.pop(
            "optimizer", torch.optim.Adam(self.parameters(), lr=lr)
        )
        self.batch_size = kwargs.pop("batch_size", 32)

        scheduler: bool = kwargs.pop("scheduler", True)

        self.scheduler: ReduceLROnPlateau = (
            ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.8, patience=5
            )  # verbose=True
            if scheduler
            else None
        )

    def _get_dataloaders(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        shuffle: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:

        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()

        X_train = (
            torch.tensor(X_train, dtype=torch.float32)
            if not isinstance(X_train, Tensor)
            else X_train.float()
        )

        if hasattr(X_val, "toarray"):
            X_val = X_val.toarray()
        X_val = (
            torch.tensor(X_val, dtype=torch.float32)
            if not isinstance(X_val, Tensor)
            else X_val.float()
        )

        if not isinstance(y_train, Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32)
        else:
            y_train = y_train.float()
        y_train = y_train.round().to(torch.int64) - 1

        if not isinstance(y_val, Tensor):
            y_val = torch.tensor(y_val, dtype=torch.float32)
        else:
            y_val = y_val.float()
        y_val = y_val.round().to(torch.int64) - 1

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
    ) -> None:

        self.to(self.device)
        patience_counter = 0
        best_val_loss = float("inf")

        train_loader, val_loader = self._get_dataloaders(X_train, y_train, X_val, y_val)

        for epoch in range(self.start_epoch, self.epochs):
            torch.cuda.empty_cache()
            self.train()
            train_loss = 0.0

            train_loss = self._train_loop(train_loader, epoch)

            train_loss = train_loss / len(train_loader)

            self.save(self.model_path / "last_model.pt", epoch)

            self.eval()

            all_preds, all_labels, val_loss = self._val_loop(val_loader)

            val_loss /= len(val_loader)

            all_preds: np.ndarray = np.concatenate(all_preds, axis=0)
            # all_labels: np.ndarray = np.array(all_labels, dtype=int)

            if all(
                isinstance(label, str) for label in all_labels[:5]
            ):  # Check if labels are strings
                all_labels = all_labels  # Keep as list of strings
            else:
                all_labels = np.array(all_labels, dtype=int)

            print(
                f"Epoch {epoch+1}/{self.epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}"
            )

            metrics: Dict[str, float] = self.evaluate(
                X=None, y=all_labels, y_pred=all_preds
            )

            print(
                "Val Metrics — ",
                ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
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
                self.save(self.model_path / "best_model.pt", epoch)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
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

    def save(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.start_epoch = state["epoch"] + 1
        self.to(self.device)

    @abstractmethod
    def _train_loop(self, train_loader: DataLoader, epoch: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def _val_loop(
        self, val_loader: DataLoader
    ) -> Tuple[List[np.ndarray], List[int], float]:
        raise NotImplementedError
