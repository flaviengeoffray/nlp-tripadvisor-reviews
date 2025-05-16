from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base_pytorch import BaseTorchModel
from models.classification.base import BaseClassificationModel


class RNNClassifier(BaseTorchModel, BaseClassificationModel):

    def __init__(
        self,
        model_path: Path,
        **kwargs: Any,
    ) -> None:

        nn.Module.__init__(self)

        input_dim: int = kwargs.pop("input_dim", 100)
        hidden_size: int = kwargs.pop("hidden_size", 128)
        num_layers: int = kwargs.pop("num_layers", 1)
        bidirectional: bool = kwargs.pop("bidirectional", False)
        dropout_rate: float = kwargs.pop("dropout_rate", 0.0)
        output_dim: int = kwargs.pop("output_dim", 5)

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * factor, output_dim)

        super().__init__(model_path=model_path, **kwargs)

    def forward(self, x: Tensor) -> Tensor:

        if x.dim() == 2:
            x = x.unsqueeze(1)
        outputs, (h_n, c_n) = self.rnn(x)

        if self.rnn.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]

        logits = self.fc(last_hidden)  # (batch_size, output_dim)

        return logits

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:

        if hasattr(X, "toarray"):
            X = X.toarray()

        self.eval()
        X = (
            torch.tensor(X, dtype=torch.float)
            if not isinstance(X, Tensor)
            else X.float()
        ).to(self.device)

        with torch.no_grad():
            outputs = self.forward(X)
            preds = torch.argmax(outputs, dim=1)

        return preds.cpu().numpy() + 1

    def _train_loop(self, train_loader: DataLoader, epoch: int) -> float:

        self.train()
        train_loss = 0.0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}"):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.forward(X)

            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(train_loader)

    def _val_loop(
        self, val_loader: DataLoader
    ) -> Tuple[List[np.ndarray], List[int], float]:

        self.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in val_loader:

                X, y = X.to(self.device), y.to(self.device)
                outputs = self.forward(X)

                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.extend(y.cpu().numpy().tolist())

                val_loss += self.criterion(outputs, y).item()

        return all_preds, all_labels, val_loss / len(val_loader)
