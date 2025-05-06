from pathlib import Path
from typing import Dict, List, Union, Any

import numpy as np
import torch
from torch import Tensor
from torch import nn

from models.base_pytorch import BaseTorchModel
from models.classification.base import BaseClassificationModel

class LSTMModel(BaseTorchModel, BaseClassificationModel):
    def __init__(
        self,
        model_path: Path,
        **kwargs: Any,
    ):
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
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

        super().__init__(model_path=model_path, kwargs=kwargs)

    def forward(self, X: Tensor) -> Tensor:
        # LSTM layers in PyTorch expect 3D input tensors
        if X.dim() == 2:
            X = X.unsqueeze(1)  # [batch_size, 1000, 1]


        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            X.size(0),
            self.hidden_dim,
            device=X.device
        )
        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(X, (h0, c0))
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
