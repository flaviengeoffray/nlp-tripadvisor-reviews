import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# from torch.utils.tensorboard import SummaryWriter

from data.datasets.TripAdvisorDataset import TripAdvisorDataset, causal_mask

from models.base_pytorch import BaseTorchModel
from models.generative.base import BaseGenerativeModel


class Embedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        self.d_model: int = d_model
        self.embedding: nn.Module = nn.Embedding(vocab_size, d_model)

    def forward(self, X: Tensor) -> Tensor:
        return self.embedding(X) * math.sqrt(self.d_model)


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
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
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
        return self.l2(self.dropout(torch.relu(self.l1(X))))  # (B, seq_len, d_model)


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

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor]
    ) -> Tensor:
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
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor],
        dropout: Optional[nn.Dropout],
    ) -> Tuple[Tensor, Tensor]:

        d_k = Q.shape[-1]

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, seq_len, seq_len)

        # Mask some values by a really small value to get the output of softmax near 0 for those values.
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        scores = scores.softmax(dim=-1)  # (B, h, seq_len, seq_len)
        if dropout is not None:
            scores = dropout(scores)

        return scores @ V, scores  # (B, h, seq_len, d_k), (B, h, seq_len, seq_len)


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()

        self.dropout: nn.Module = nn.Dropout(dropout)
        self.norm: LayerNorm = LayerNorm()

    def forward(self, X: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        # Most of online implementations online are doing the layer norm before the sublayer.
        return X + self.dropout(sublayer(self.norm(X)))


class EncoderBlock(nn.Module):

    def __init__(
        self,
        self_attention: MultiHeadAttention,
        feedforward: FeedForward,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attention: MultiHeadAttention = self_attention
        self.res_self_attention: ResidualConnection = ResidualConnection(dropout)
        self.feedforward: FeedForward = feedforward
        self.res_feedforward: ResidualConnection = ResidualConnection(dropout)

    def forward(self, X, source_mask: Tensor) -> Tensor:

        X = self.res_self_attention(
            X, lambda X: self.self_attention(X, X, X, source_mask)
        )
        X = self.res_feedforward(X, self.feedforward)

        return X


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers: nn.ModuleList[EncoderBlock] = layers
        self.norm: LayerNorm = LayerNorm()

    def forward(self, X: Tensor, source_mask: Tensor) -> Tensor:

        for layer in self.layers:
            X = layer(X, source_mask)

        return self.norm(X)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feedforward: FeedForward,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attention: MultiHeadAttention = self_attention
        self.res_self_attention: ResidualConnection = ResidualConnection(dropout)
        self.cross_attention: MultiHeadAttention = cross_attention
        self.res_cross_attention: ResidualConnection = ResidualConnection(dropout)
        self.feedforward: FeedForward = feedforward
        self.res_feedforward: ResidualConnection = ResidualConnection(dropout)

    def forward(
        self,
        X: Tensor,
        encoder_output: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:

        X = self.res_self_attention(
            X, lambda X: self.self_attention(X, X, X, target_mask)
        )

        X = self.res_cross_attention(
            X,
            lambda X: self.cross_attention(
                X, encoder_output, encoder_output, source_mask
            ),
        )

        X = self.res_feedforward(X, self.feedforward)

        return X


class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers: nn.ModuleList[DecoderBlock] = layers
        self.norm: LayerNorm = LayerNorm()

    def forward(
        self,
        X: Tensor,
        encoder_output: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:

        for layer in self.layers:
            X = layer(X, encoder_output, source_mask, target_mask)

        return self.norm(X)


class Projection(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        self.linear: nn.Module = nn.Linear(d_model, vocab_size)

    def forward(self, X: Tensor) -> Tensor:
        # Log-Softmax is used here for numerical stability.
        return torch.log_softmax(self.linear(X), dim=-1)  # (B, seq_len, vocab_sizs)


# class TransformerBlock(nn.Module):

#     def __init__(
#         self,
#         encoder: Encoder,
#         decoder: Decoder,
#         source_embedding: Embedding,
#         target_embedding: Embedding,
#         source_position: PositionalEncoding,
#         target_position: PositionalEncoding,
#         projection: Projection,
#     ) -> None:
#         super().__init__()

#         self.encoder: Encoder = encoder
#         self.decoder: Decoder = decoder
#         self.source_embedding: Embedding = source_embedding
#         self.target_embedding: Embedding = target_embedding
#         self.source_position: PositionalEncoding = source_position
#         self.target_position: PositionalEncoding = target_position
#         self.projection: Projection = projection

#     def encode(self, source: Tensor, source_mask: Tensor) -> Tensor:

#         source = self.source_embedding(source)
#         source = self.source_position(source)

#         return self.encoder(source)

#     def decode(
#         self,
#         encoder_output: Tensor,
#         source_mask: Tensor,
#         target: Tensor,
#         target_mask: Tensor,
#     ) -> Tensor:

#         target = self.target_embedding(target)
#         target = self.target_position(target)

#         return self.decoder(target, encoder_output, source_mask, target_mask)

#     def project(self, X: Tensor) -> Tensor:
#         return self.projection(X)


class Transformer(BaseTorchModel, BaseGenerativeModel):

    def __init__(self, model_path, **kwargs):
        # super().__init__(model_path, **kwargs)
        nn.Module.__init__(self)

        self.vocab_size: int = kwargs.pop("vocab_size", 30000)
        self.d_model: int = kwargs.pop("d_model", 512)
        self.N: int = kwargs.pop("N", 6)
        self.h: int = kwargs.pop("h", 8)
        self.dropout: float = kwargs.pop("dropout", 0.1)
        self.d_ff: int = kwargs.pop("d_ff", 2048)
        self.max_target_len: int = kwargs.pop("max_target_len", 256)

        self.max_input_len: int = kwargs.pop("max_input_len", 32)
        # self.tokenizer: BaseTokenizer = kwargs.pop("tokenizer", BpeTokenizer())

        # Embeddings Layers
        self.input_embedding: Embedding = Embedding(self.d_model, self.vocab_size)
        self.output_embedding: Embedding = Embedding(self.d_model, self.vocab_size)

        # Positional Encoding Layers
        self.source_position: PositionalEncoding = PositionalEncoding(
            self.d_model, self.max_target_len, self.dropout
        )
        self.target_position: PositionalEncoding = PositionalEncoding(
            self.d_model, self.max_target_len, self.dropout
        )

        # Encoder layer
        encoder_blocks = [
            EncoderBlock(
                MultiHeadAttention(self.d_model, self.h, self.dropout),
                FeedForward(self.d_model, self.d_ff, self.dropout),
                self.dropout,
            )
            for _ in range(self.N)
        ]
        self.encoder: Encoder = Encoder(nn.ModuleList(encoder_blocks))

        # Decoder layer
        decoder_blocks = [
            DecoderBlock(
                MultiHeadAttention(self.d_model, self.h, self.dropout),
                MultiHeadAttention(self.d_model, self.h, self.dropout),
                FeedForward(self.d_model, self.d_ff, self.dropout),
                self.dropout,
            )
            for _ in range(self.N)
        ]

        self.decoder: Decoder = Decoder(nn.ModuleList(decoder_blocks))

        # Projection layer
        self.projection: Projection = Projection(self.d_model, self.vocab_size)

        # TODO: init params with xavier_uniform

        super().__init__(model_path=model_path, **kwargs)

        assert self.tokenizer is not None, "Transformer model must have a tokenizer."

        self.criterion: nn.Module = kwargs.pop(
            "criterion",
            nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.token_to_id("[PAD]"), label_smoothing=0.1
            ),
        ).to(self.device)

    def encode(self, source: Tensor, source_mask: Tensor) -> Tensor:

        source = self.input_embedding(source)
        source = self.source_position(source)

        return self.encoder(source, source_mask)

    def decode(
        self,
        encoder_output: Tensor,
        source_mask: Tensor,
        target: Tensor,
        target_mask: Tensor,
    ) -> Tensor:

        target = self.output_embedding(target)
        target = self.target_position(target)

        return self.decoder(target, encoder_output, source_mask, target_mask)

    def project(self, X: Tensor) -> Tensor:
        return self.projection(X)

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

        train_ds = TripAdvisorDataset(
            X_train.tolist(),
            y_train.tolist(),
            self.tokenizer,
            self.max_input_len,
            self.max_target_len,
        )

        if not hasattr(X_val, "tolist") or not hasattr(y_val, "tolist"):
            raise ValueError("X_val and y_val needs to by numpy arrays")

        val_ds = TripAdvisorDataset(
            X_val.tolist(),
            y_val.tolist(),
            self.tokenizer,
            self.max_input_len,
            self.max_target_len,
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader

    def inference(self, encoder_input: Tensor, encoder_mask: Tensor) -> Tensor:

        sos = self.tokenizer.token_to_id("[SOS]")
        eos = self.tokenizer.token_to_id("[EOS]")

        encoder_output: Tensor = self.encode(encoder_input, encoder_mask)

        decoder_input: Tensor = (
            torch.empty(1, 1).fill_(sos).type_as(encoder_input).to(self.device)
        )

        while True:
            if decoder_input.size(1) == self.max_target_len:
                break

            decoder_mask = (
                causal_mask(decoder_input.size(1))
                .type_as(encoder_input)
                .to(self.device)
            )

            decoder_output = self.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )

            # This is a greedy decoding since we always take the max probable token
            # next_token = torch.max(self.project(decoder_output[:, -1]), dim=-1)[1]

            # Topk decoding
            topk = 10
            logits = self.project(decoder_output[:, -1])
            topk_logits, topk_indices = torch.topk(logits, topk)
            probs = torch.softmax(topk_logits, dim=-1)
            next_token = topk_indices[0, torch.multinomial(probs, num_samples=1)]

            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1)
                    .type_as(encoder_input)
                    .fill_(next_token.item())
                    .to(self.device),
                ],
                dim=1,
            )

            if next_token == eos:
                break

        return decoder_input.squeeze(0)

    def generate(self, prompt: Optional[Any] = None) -> Any:
        """
        Generate a sentence from an input data.
        """
        self.eval()

        sos: Tensor = torch.tensor(
            [self.tokenizer.token_to_id("[SOS]")], dtype=torch.int64
        )
        eos: Tensor = torch.tensor(
            [self.tokenizer.token_to_id("[EOS]")], dtype=torch.int64
        )
        pad: Tensor = torch.tensor(
            [self.tokenizer.token_to_id("[PAD]")], dtype=torch.int64
        )

        source_tokens = self.tokenizer.encode(prompt)[: self.max_input_len]

        source_padding_len = self.max_target_len - len(source_tokens) - 2

        if source_padding_len < 0:
            raise ValueError("Sentence length error")

        encoder_input: Tensor = torch.cat(
            [
                sos,
                torch.tensor(source_tokens, dtype=torch.int64),
                eos,
                torch.tensor([pad] * source_padding_len, dtype=torch.int64),
            ]
        ).unsqueeze(0)
        encoder_mask: Tensor = (encoder_input != pad).unsqueeze(1).unsqueeze(2).int()

        with torch.no_grad():
            output = self.inference(encoder_input, encoder_mask)

        return self.tokenizer.decode(output.detach().cpu().tolist())

    def _train_loop(self, train_loader: DataLoader, epoch: int) -> float:

        train_loss: float = 0.0

        for B in tqdm(train_loader, desc=f"Processing epoch: {epoch}/{self.epochs}"):

            self.optimizer.zero_grad()

            encoder_input: Tensor = B["encoder_input"].to(self.device)  # (B, seq_len)
            decoder_input: Tensor = B["decoder_input"].to(self.device)  # (B, seq_len)
            encoder_mask: Tensor = B["encoder_mask"].to(
                self.device
            )  # (B, 1, 1, seq_len)
            decoder_mask: Tensor = B["decoder_mask"].to(
                self.device
            )  # (B, 1, seq_len, seq_len)

            encoder_output: Tensor = self.encode(
                encoder_input, encoder_mask
            )  # (B, seq_len, d_model)
            decoder_output: Tensor = self.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (B, seq_len, d_model)

            projection_output: Tensor = self.project(
                decoder_output
            )  # (B, seq_len, vocab_size)

            label: Tensor = B["label"].to(self.device)  # (B, seq_len)

            loss: Tensor = self.criterion(
                projection_output.view(-1, self.vocab_size), label.view(-1)
            )

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss

    def _val_loop(
        self, val_loader: DataLoader
    ) -> Tuple[List[np.ndarray], List[int], float]:

        source_texts = []
        expected_texts = []
        preds = []

        val_loss: float = 0.0

        with torch.no_grad():
            i = 0
            for B in val_loader:
                encoder_input: Tensor = B["encoder_input"].to(
                    self.device
                )  # (B, seq_len)
                encoder_mask: Tensor = B["encoder_mask"].to(
                    self.device
                )  # (B, 1, 1, seq_len)

                label: Tensor = B["label"].to(self.device)  # (B, seq_len)

                # assert (
                #     encoder_input.size(0) == 1
                # ), "Validation batch size must be equals to 1."

                output: Tensor = self.inference(encoder_input, encoder_mask)

                # loss: Tensor = self.criterion(
                #     projection_output.view(-1, self.vocab_size), label.view(-1)
                # )

                output_text = self.tokenizer.decode(output.detach().cpu().tolist())

                source_text = B["source_text"][0]
                target_text = B["target_text"][0]

                source_texts.append(source_text)
                expected_texts.append(target_text)
                preds.append(output_text)

                if i % 100 == 0:
                    print(f"SOURCE: {source_text}")
                    print(f"TARGET: {target_text}")
                    print(f"PREDICTED: {output_text}")
                i += 1

        return preds, expected_texts, val_loss
