import math
from typing import Callable, Tuple
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
        self.alpha: nn.Module = nn.Parameter(torch.ones(1))
        self.bias: nn.Module = nn.Parameter(torch.zeros(1))

    def forward(self, X: Tensor) -> Tensor:

        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)

        return self.alpha * ((X - mean) / (std + self.epsilon)) + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        # self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        # self.relu = nn.ReLU()
        # self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        self.l1: nn.Module = nn.Linear(d_model, d_ff)
        self.dropout: nn.Module = nn.Dropout(dropout)
        self.l2: nn.Module = nn.Linear(d_ff, d_model)

    def forward(self, X: Tensor) -> Tensor:
        return self.l2(self.dropout(torch.relu(self.l2(X))))  # (B, seq_len, d_model)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: 0.1) -> None:
        super().__init__()

        self.d_model: int = d_model
        self.h: int = h
        self.dropout: nn.Module = nn.Dropout(dropout)

        assert d_model % h == 0, "d_model must be divisible by h."

        self.d_k: int = d_model // h

        # Init Weights
        self.w_q: nn.Module = nn.Linear(d_model, d_model)  # Query
        self.w_k: nn.Module = nn.Linear(d_model, d_model)  # Key
        self.w_v: nn.Module = nn.Linear(d_model, d_model)  # Value

        self.w_o: nn.Module = nn.Linear(d_model, d_model)  # Output

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor) -> Tensor:
        Q = self.w_q(Q)  # (B, seq_len, d_model)
        K = self.w_k(K)  # (B, seq_len, d_model)
        V = self.w_v(V)  # (B, seq_len, d_model)

        Q = Q.view(Q.shape[0], Q.shape[1], self.h, self.d_k).transpose(
            1, 2
        )  # (B, h, seq_len, d_k)
        K = K.view(K.shape[0], K.shape[1], self.h, self.d_k).transpose(
            1, 2
        )  # (B, h, seq_len, d_k)
        V = V.view(V.shape[0], V.shape[1], self.h, self.d_k).transpose(
            1, 2
        )  # (B, h, seq_len, d_k)

        X, self.scores = MultiHeadAttention.attention(Q, K, V, mask, self.dropout)

        X = (
            X.transpose(1, 2).contiguous().view(X.shape[0], -1, self.d_model)
        )  # (B, seq_len, d_model)

        return self.w_o(X)  # (B, seq_len, d_model)

    @staticmethod
    def attention(
        Q: Tensor, K: Tensor, V: Tensor, mask: Tensor, dropout: nn.Dropout
    ) -> Tuple[Tensor, Tensor]:

        d_k = Q.shape[-1]

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, seq_len, seq_len)

        # Mask some values by a really small value to get the output of softmax near 0 for those values.
        if mask:
            scores.masked_fill_(mask == 0, -1e9)

        scores = scores.softmax(dim=-1)  # (B, h, seq_len, seq_len)
        if dropout:
            scores = dropout(scores)

        return scores @ V, scores  # (B, h, seq_len, d_k), (B, h, seq_len, seq_len)


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()

        self.dropout: nn.Module = nn.Dropout(dropout)
        self.norm: nn.Module = LayerNorm()

    def forward(self, X: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        # Most of online implementations online are doing the layer norm before the sublayer.
        return X + self.dropout(sublayer(self.norm(X)))


class EncoderBlock(nn.Module):

    def __init__(
        self,
        attention: MultiHeadAttention,
        feedforward: FeedForward,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention: MultiHeadAttention = attention
        self.res_attention: ResidualConnection = ResidualConnection(dropout)
        self.feedforward: FeedForward = feedforward
        self.res_feedforward: ResidualConnection = ResidualConnection(dropout)

    def forward(self, X, mask: Tensor) -> None:

        X = self.res_attention(X, lambda X: self.attention(X, X, X, mask))
        X = self.res_feedforward(X, self.feedforward)

        return X


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        self.layers: nn.ModuleList[EncoderBlock] = layers
        self.norm: LayerNorm = LayerNorm()

    def forward(self, X: Tensor, mask: Tensor) -> Tensor:

        for layer in self.layers:
            X = layer(X, mask)

        return self.norm(X)


class Transformer(BaseTorchModel, BaseGenerativeModel):

    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)

        self.vocab_size: int = kwargs.pop("vocab_size", 10000)
        self.d_model: int = kwargs.pop("d_model", 128)

        self.input_embedding = nn.Embedding(
            self.vocab_size, self.d_model
        )  # Maybe need to mult by srqt of d_model
