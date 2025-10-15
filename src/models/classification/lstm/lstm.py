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
from data.text_dataset import TextDataset


class LSTMModel(BaseTorchModel, BaseClassificationModel):
    """This implementation may not be working"""

    def __init__(
        self,
        model_path: Path,
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)

        input_dim: int = kwargs.pop("input_dim", 1)
        hidden_dim: int = kwargs.pop("hidden_dim", 128)
        output_dim: int = kwargs.pop("output_dim", 5)
        num_layers: int = kwargs.pop("num_layers", 1)
        dropout_rate: float = kwargs.pop("dropout_rate", 0.0)
        bidirectional: bool = kwargs.pop("bidirectional", True)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

        super().__init__(model_path=model_path, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        X = self.vectorizer.transform(X)  # Transform input texts to vectors

        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            X.size(0),
            self.hidden_dim,
            device=X.device,
        )
        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(X, (h0, c0))  # out: [batch_size, seq_len, hidden_dim*directions]
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        self.model.eval()
        X = (
            torch.tensor(X, dtype=torch.float32)
            if not isinstance(X, Tensor)
            else X.float()
        ).to(self.device)

        with torch.no_grad():
            outputs = self.forward(X)
            preds = torch.argmax(outputs, dim=1)
        return preds.cpu().numpy()

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

        if hasattr(X_val, "toarray"):
            X_val = X_val.toarray()

        if not hasattr(X_train, "tolist") or not hasattr(y_train, "tolist"):
            raise ValueError("X_train and y_train needs to by numpy arrays")

        train_ds = TextDataset(
            texts=X_train.tolist(),
            labels=y_train.tolist(),
            tokenizer=self.tokenizer,
            max_len=512,
        )

        if not hasattr(X_val, "tolist") or not hasattr(y_val, "tolist"):
            raise ValueError("X_val and y_val needs to by numpy arrays")

        val_ds = TextDataset(
            texts=X_val.tolist(),
            labels=y_val.tolist(),
            tokenizer=self.tokenizer,
            max_len=512,
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader

    def _train_loop(self, train_loader: DataLoader, epoch: int) -> float:
        train_loss: float = 0.0
        for B in tqdm(train_loader, desc=f"Processing epoch: {epoch}/{self.epochs}"):
            # X, y = X.to(self.device), y.to(self.device)
            X = B["input_ids"].to(self.device)
            y = B["label"].to(self.device)
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
            for B in tqdm(val_loader, desc="Processing validation"):
                X = B["input_ids"].to(self.device)
                y = B["label"].to(self.device)
                outputs = self.forward(X)

                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.extend(y.cpu().numpy().tolist())

                loss = self.criterion(outputs, y)
                val_loss += loss.item()

        return all_preds, all_labels, val_loss
