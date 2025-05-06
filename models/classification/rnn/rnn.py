from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data.datasets.TextDataset import TextDataset
from models.base_pytorch import BaseTorchModel
from models.classification.base import BaseClassificationModel


class RNNModel(BaseTorchModel, BaseClassificationModel):
    def __init__(
        self,
        model_path: Path,
        **kwargs: Any,
    ):
        nn.Module.__init__(self)

        vocab_size: int = kwargs.pop("vocab_size", 10000)
        input_dim: int = kwargs.pop("input_dim", 256)
        hidden_size: int = kwargs.pop("hidden_dims", 64)
        output_size: int = kwargs.pop("output_size", 5)
        dropout_rate: float = kwargs.pop("dropout_rate", None)
        num_layers: int = kwargs.pop("num_layers", 1)

        # self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(vocab_size, input_dim)

        self.rnn = nn.RNN(input_dim, hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        super().__init__(model_path=model_path, **kwargs)  # =kwargs)

    # def forward(self, x: Tensor) -> Tensor:

    #     # x = self.flatten(x)
    #     # if x.dim() == 2:
    #     #     x = x.unsqueeze(0)
    #     # out = x
    #     print(x.shape)
    #     out = self.embeddings(x)
    #     print(out.shape)
    #     rnn_out, hidden = self.rnn(out)
    #     print(hidden.shape)

    #     hidden = self.dropout(hidden)
    #     output = self.h2o(hidden)
    #     print(output.shape)

    #     output = self.softmax(output)

    #     return output

    def forward(self, x: Tensor, text_lengths: Optional[Tensor] = None) -> Tensor:
        # x: [batch, seq_len]  (dtype=torch.long)

        embedded = self.embedding(x)
        # embedded: [batch, seq_len, embed_dim]

        # Pack/Unpack si vous voulez gérer le padding
        if text_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, hidden = self.rnn(packed)
        else:
            output, hidden = self.rnn(embedded)
        # hidden: [num_layers * num_directions, batch, hidden_dim]

        # Prenez la dernière couche cachée
        # Pour un RNN unidirectionnel à 1 layer :
        h = hidden[-1]
        # h: [batch, hidden_dim]

        h = self.dropout(h)
        logits = self.fc(h)
        # logits: [batch, output_dim]

        return self.softmax(logits)

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        self.eval()
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

        if not hasattr(X_train, "tolist") or not hasattr(y_train, "tolist"):
            raise ValueError("X_train and y_train needs to by numpy arrays")

        # if nmo

        train_ds = TextDataset(
            texts=X_train.tolist(),
            labels=y_train.tolist(),
            tokenizer=self.tokenizer,
            max_len=128,
        )

        if not hasattr(X_val, "tolist") or not hasattr(y_val, "tolist"):
            raise ValueError("X_val and y_val needs to by numpy arrays")

        val_ds = TextDataset(
            texts=X_val.tolist(),
            labels=y_val.tolist(),
            tokenizer=self.tokenizer,
            max_len=128,
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader

    def _train_loop(self, train_loader: DataLoader, epoch: int) -> float:
        train_loss: float = 0.0
        for B in tqdm(train_loader, desc=f"Processing epoch: {epoch}/{self.epochs}"):
            X: Tensor = B["input_ids"].to(self.device)
            y: Tensor = B["label"].to(self.device)
            length: Tensor = B["length"].to(self.device)
            # y = y.view(y.size(0), 1)
            self.optimizer.zero_grad()
            outputs = self.forward(X, length)

            # outputs = torch.argmax(outputs, dim=1)

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
            for B in val_loader:
                # X, y = X.to(self.device), y.to(self.device)
                X: Tensor = B["input_ids"].to(self.device)
                y: Tensor = B["label"].to(self.device)
                length: Tensor = B["length"].to(self.device)
                outputs = self.forward(X, length)

                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.extend(y.cpu().numpy().tolist())

                loss = self.criterion(outputs, y)
                val_loss += loss.item()

        return all_preds, all_labels, val_loss
