from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from gensim.models import Word2Vec

from models.base_pytorch import BaseTorchModel
from models.classification.base import BaseClassificationModel
from vectorizers.word2vec import Word2VecVectorizer


class RNNClassifier(BaseTorchModel, BaseClassificationModel):
    def __init__(
        self,
        model_path: Path,
        **kwargs: Any,
    ):
        # Initialize nn.Module
        nn.Module.__init__(self)

        # RNN parameters
        input_dim: int = kwargs.pop("input_dim", 100)
        hidden_size: int = kwargs.pop("hidden_size", 128)
        num_layers: int = kwargs.pop("num_layers", 1)
        bidirectional: bool = kwargs.pop("bidirectional", False)
        dropout_rate: float = kwargs.pop("dropout_rate", 0.0)
        output_dim: int = kwargs.pop("output_dim", 5)  # Number of classes

        # Define RNN
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Final classification layer
        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * factor, output_dim)

        # Save forward signature
        super().__init__(model_path=model_path, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        outputs, (h_n, c_n) = self.rnn(x)

        # Use the last hidden state
        if self.rnn.bidirectional:
            # For bidirectional, concatenate the last hidden state from both directions
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            # For unidirectional, just use the last hidden state
            last_hidden = h_n[-1]

        # Pass through the final linear layer to get class logits
        logits = self.fc(last_hidden)  # shape: (batch_size, output_dim)
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
                # Make sure y is the right shape
                # y = y.squeeze() if y.dim() > 1 else y

                X, y = X.to(self.device), y.to(self.device)
                outputs = self.forward(X)
                # _, preds = torch.max(outputs, dim=1)

                # Store predictions and labels
                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.extend(y.cpu().numpy().tolist())

                # Compute loss
                val_loss += self.criterion(outputs, y).item()

        # Concatenate all predictions
        # all_preds = np.concatenate(all_preds) if all_preds else np.array([])

        return all_preds, all_labels, val_loss / len(val_loader)
