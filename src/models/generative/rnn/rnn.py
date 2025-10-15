import numpy as np
from pathlib import Path
from typing import List, Any, Tuple
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.tripadvisor_dataset import TripAdvisorDataset

from models.base_pytorch import BaseTorchModel
from models.generative.base import BaseGenerativeModel


class RNNGenModel(BaseTorchModel, BaseGenerativeModel):
    def __init__(self, model_path, **kwargs) -> None:
        nn.Module.__init__(self)

        self.vocab_size: int = kwargs.pop("vocab_size", 5000)
        self.embedding_dim: int = kwargs.pop("embedding_dim", 256)
        self.hidden_dim = kwargs.pop("hidden_dim", 512)
        self.num_layers = kwargs.pop("num_layers", 2)
        self.dropout_rate = kwargs.pop("dropout", 0.3)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)

        BaseTorchModel.__init__(self, model_path=model_path, **kwargs)
        BaseGenerativeModel.__init__(self, model_path)

    def forward(self, x, hidden=None) -> Tensor:
        batch_size = x.size(0)

        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
                self.device
            )
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
                self.device
            )
            hidden = (h0, c0)

        output, hidden = self.lstm(
            embedded, hidden
        )  # (batch_size, seq_len, hidden_dim)

        output = self.dropout(output)

        output = self.fc(output)  # (batch_size, seq_len, vocab_size)

        return output

    def _get_dataloaders(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        shuffle: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        train_dataset = TripAdvisorDataset(
            texts=X_train,
            ratings=y_train,
            tokenizer=self.tokenizer,
            max_input_len=64,
            max_target_len=256,
        )

        val_dataset = TripAdvisorDataset(
            texts=X_val,
            ratings=y_val,
            tokenizer=self.tokenizer,
            max_input_len=64,
            max_target_len=256,
        )

        # Define collate function to handle variable length sequences
        def collate_fn(batch):
            batch_data = {
                "encoder_input": [],
                "decoder_input": [],
                "encoder_mask": [],
                "decoder_mask": [],
                "label": [],
                "source_text": [],
                "target_text": [],
            }

            for item in batch:
                for key in batch_data:
                    batch_data[key].append(item[key])

            for key in [
                "encoder_input",
                "decoder_input",
                "encoder_mask",
                "decoder_mask",
                "label",
            ]:
                batch_data[key] = torch.stack(batch_data[key])

            return batch_data

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader

    def _train_loop(self, train_loader: DataLoader, epoch: int) -> float:
        self.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
            input_seq = batch["decoder_input"].to(self.device)
            target_seq = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            output, _ = self.forward(input_seq)

            output = output.reshape(
                -1, self.vocab_size
            )  # (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            target_seq = target_seq.reshape(
                -1
            )  # (batch_size, seq_len) -> (batch_size * seq_len)

            # Create a mask to ignore padding tokens in loss calculation
            # -1 for ignored positions (PAD tokens)
            pad_mask = target_seq != self.tokenizer.token_to_id("[PAD]")

            masked_target = target_seq[pad_mask]
            masked_output = output[pad_mask]

            loss = self.criterion(masked_output, masked_target)

            loss.backward()

            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss

    def _val_loop(
        self, val_loader: DataLoader
    ) -> Tuple[List[np.ndarray], List[str], float]:
        self.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_seq = batch["decoder_input"].to(self.device)
                target_seq = batch["label"].to(self.device)

                output = self.forward(input_seq)

                _, preds = torch.max(output, dim=2)

                output_flat = output.reshape(-1, self.vocab_size)
                target_flat = target_seq.reshape(-1)
                pad_mask = target_flat != self.tokenizer.token_to_id("[PAD]")
                loss = self.criterion(output_flat[pad_mask], target_flat[pad_mask])
                val_loss += loss.item()

                pred_seqs = preds.cpu().numpy()  # (batch, seq_len)
                tgt_seqs = target_seq.cpu().numpy()

                for pred_ids, tgt_ids in zip(pred_seqs, tgt_seqs):
                    eos = np.where(pred_ids == self.tokenizer.token_to_id("[EOS]"))[0]
                    if len(eos):
                        pred_ids = pred_ids[: eos[0] + 1]
                        tgt_ids = tgt_ids[: eos[0] + 1]

                    pred_tokens = [
                        t
                        for t in pred_ids
                        if t
                        not in (
                            self.tokenizer.token_to_id("[PAD]"),
                            self.tokenizer.token_to_id("[SOS]"),
                        )
                    ]
                    tgt_tokens = [
                        t
                        for t in tgt_ids
                        if t
                        not in (
                            self.tokenizer.token_to_id("[PAD]"),
                            self.tokenizer.token_to_id("[SOS]"),
                        )
                    ]

                    all_preds.append(self.tokenizer.decode(pred_tokens))
                    all_labels.append(self.tokenizer.decode(tgt_tokens))

        idxs = list(range(0, len(all_preds), 100))
        all_preds = [all_preds[i] for i in idxs]
        all_labels = [all_labels[i] for i in idxs]

        return [all_preds], all_labels, val_loss

    def generate(
        self,
        rating: int,
        keywords: str,
        max_length: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        self.eval()

        prompt = f"{rating}: {keywords}"
        prompt_tokens = self.tokenizer.encode(prompt)

        input_tokens = [self.tokenizer.token_to_id("[SOS]")] + prompt_tokens
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(self.device)

        generated_tokens = []
        hidden = None

        min_length = 30  # At least 30 tokens before EOS

        with torch.no_grad():
            output, hidden = self.forward(input_tensor, hidden)
            current_token = input_tensor[:, -1].unsqueeze(1)  # Last token

            # Generate tokens one by one
            for i in range(max_length):
                output, hidden = self.forward(current_token, hidden)
                next_token_logits = output[0, -1, :] / temperature

                # Block EOS and PAD tokens if the minimum length has not been reached
                if i < min_length:
                    eos_id = self.tokenizer.token_to_id("[EOS]")
                    pad_id = self.tokenizer.token_to_id("[PAD]")
                    next_token_logits[eos_id] = float("-inf")
                    next_token_logits[pad_id] = float("-inf")

                # Apply top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(
                        next_token_logits, float("-inf")
                    )
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)

                # Apply top-p sampling
                if top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # Stop if EOS or PAD and minimum length is reached
                if i >= min_length and next_token in [
                    self.tokenizer.token_to_id("[EOS]"),
                    self.tokenizer.token_to_id("[PAD]"),
                ]:
                    break

                generated_tokens.append(next_token)
                current_token = torch.tensor([[next_token]], dtype=torch.long).to(
                    self.device
                )

        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text

    def save(self, path: Path, epoch: int = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch if epoch is not None else 0,
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout_rate,
            },
            path,
        )

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.start_epoch = state.get("epoch", 0) + 1

        if "vocab_size" in state:
            self.vocab_size = state["vocab_size"]
        if "embedding_dim" in state:
            self.embedding_dim = state["embedding_dim"]
        if "hidden_dim" in state:
            self.hidden_dim = state["hidden_dim"]
        if "num_layers" in state:
            self.num_layers = state["num_layers"]
        if "dropout" in state:
            self.dropout_rate = state["dropout"]
