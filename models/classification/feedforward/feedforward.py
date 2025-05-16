from pathlib import Path
from typing import List, Tuple, Union, Any

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base_pytorch import BaseTorchModel
from models.classification.base import BaseClassificationModel


class FNNModel(BaseTorchModel, BaseClassificationModel):

    def __init__(
        self,
        model_path: Path,
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)

        input_dim: int = kwargs.pop("input_dim", 1)
        hidden_dims: List[int] = kwargs.pop("hidden_dims", [128])
        output_dim: int = kwargs.pop("output_dim", 5)
        dropout_rate: float = kwargs.pop("dropout_rate", None)

        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(dims[-1], output_dim))

        self.layers = nn.ModuleList(layers)

        super().__init__(model_path=model_path, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:

        if hasattr(X, "toarray"):
            X = X.toarray()

        self.eval()
        X = (
            torch.tensor(X, dtype=torch.float32)
            if not isinstance(X, Tensor)
            else X.float()
        ).to(self.device)

        with torch.no_grad():
            outputs = self.forward(X)
            preds = torch.argmax(outputs, dim=1)

        return preds.cpu().numpy() + 1

    def _train_loop(self, train_loader: DataLoader, epoch: int) -> float:

        train_loss: float = 0.0

        for X, y in tqdm(train_loader, desc=f"Processing epoch: {epoch}/{self.epochs}"):
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss

    def _val_loop(
        self, val_loader: DataLoader
    ) -> Tuple[List[np.ndarray], List[int], float]:
        val_loss = 0.0

        all_preds: List[np.ndarray] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.forward(X)

                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.extend(y.cpu().numpy().tolist())

                loss = self.criterion(outputs, y)
                val_loss += loss.item()

        return all_preds, all_labels, val_loss
