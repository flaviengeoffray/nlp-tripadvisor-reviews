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

        # Load pretrained Word2Vec
        w2v_model: Word2VecVectorizer = kwargs.pop("vectorizer", None)
        if w2v_model is None:
            raise Exception("Word2vec is needed for RNN model")
        emb_weights = torch.FloatTensor(w2v_model.model.wv.vectors)
        freeze_emb = kwargs.pop("freeze_embeddings", False)
        self.embedding = nn.Embedding.from_pretrained(emb_weights, freeze=freeze_emb)

        # RNN parameters
        embed_dim = emb_weights.size(1)
        hidden_size: int = kwargs.pop("hidden_size", 128)
        num_layers: int = kwargs.pop("num_layers", 1)
        bidirectional: bool = kwargs.pop("bidirectional", False)
        dropout_rate: float = kwargs.pop("dropout_rate", 0.0)
        output_dim: int = kwargs.pop("output_dim", 5)

        # Define RNN
        self.rnn = nn.LSTM(
            input_size=embed_dim,
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
        # x: LongTensor of shape (batch_size, seq_len)
        emb = self.embedding(x)  # -> (batch, seq_len, embed_dim)
        outputs, (h_n, c_n) = self.rnn(emb)
        # use last layer's hidden state
        last_hidden = h_n[-1]  # -> (batch, hidden_size * num_directions)
        logits = self.fc(last_hidden)
        return logits

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()

        self.eval()
        X = (
            torch.tensor(X, dtype=torch.long) if not isinstance(X, Tensor) else X.long()
        ).to(self.device)

        with torch.no_grad():
            outputs = self.forward(X)
            preds = torch.argmax(outputs, dim=1)

        return preds.cpu().numpy() + 1

    def _train_loop(self, train_loader: DataLoader, epoch: int) -> float:
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}"):
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
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.forward(X)

                all_preds.append(outputs.cpu().numpy())
                all_labels.extend(y.cpu().numpy().tolist())

                val_loss += self.criterion(outputs, y).item()

        return all_preds, all_labels, val_loss
