from pathlib import Path
from typing import Dict, List, Union, Any

import numpy as np
import torch
from torch import Tensor
from torch import nn

from models.base_pytorch import BaseTorchModel
from models.classification.base import BaseClassificationModel

class RNNModel(BaseTorchModel, BaseClassificationModel):
    def __init__(
        self,
        model_path: Path,
        **kwargs: Any,
    ):
        nn.Module.__init__(self)
        
        input_dim: int = kwargs.pop("input_dim", 1000)
        hidden_size: int = kwargs.pop("hidden_dims", 256)
        output_size: int = kwargs.pop("output_size", 5)
        dropout_rate: float = kwargs.pop("dropout_rate", None)
        num_layers: int = kwargs.pop("num_layers", 1)
        
        self.flatten = nn.Flatten()
        self.rnn = nn.RNN(input_dim, hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        super().__init__(model_path=model_path, kwargs=kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        out = x
        rnn_out, hidden = self.rnn(out)
        hidden = self.dropout(hidden) 
        output = self.h2o(hidden.squeeze(0)) 
        output = self.softmax(output)
        
        return output

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
