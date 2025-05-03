import math
import torch
from torch import nn
from torch import Tensor

from models.base_pytorch import BaseTorchModel
from models.generative.base import BaseGenerativeModel


# dim: int
# n_layers: int
# head_dim: int
# hidden_dim: int
# n_heads: int
# n_kv_heads: int
# norm_eps: float
# vocab_size: int

# max_batch_size: int = 0


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.d_model: int = d_model
        self.max_len: int = max_len
        self.dropout: nn.Module = nn.Dropout(dropout)

        pe: Tensor = torch.zeros(max_len, d_model)  # (max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Stock without grad for model saving.
        self.register_buffer("pe", pe)

    def forward(self, X: Tensor) -> Tensor:
        # X is (B, seq_len, d_model)
        X = X + (self.pe[:, : X.shape[1], :]).requires_grad_(False)
        return self.dropout(X)


class LayerNorm(nn.Module):

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()

        # Avoid zero division
        self.epsilon: float = epsilon

        # To allow the model to amplifies values when needed.
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: Tensor) -> Tensor:

        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)

        return self.alpha * ((X - mean) / (std + self.epsilon)) + self.bias


class Transformer(BaseTorchModel, BaseGenerativeModel):

    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)

        self.vocab_size: int = kwargs.pop("vocab_size", 10000)
        self.d_model: int = kwargs.pop("d_model", 128)

        self.input_embedding = nn.Embedding(
            self.vocab_size, self.d_model
        )  # Maybe need to mult by srqt of d_model
