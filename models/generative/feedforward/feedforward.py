import numpy as np
from pathlib import Path
from typing import List, Any, Tuple, Union
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets.TripAdvisorDataset import TripAdvisorDataset

from models.base_pytorch import BaseTorchModel
from models.generative.base import BaseGenerativeModel


class FNNGenerativeModel(BaseTorchModel, BaseGenerativeModel):

    def __init__(self, model_path: Path, **kwargs: Any) -> None:

        nn.Module.__init__(self)

        self.input_dim: int = kwargs.pop("input_dim", 1)
        self.hidden_dims: List[int] = kwargs.pop("hidden_dims", [128])
        self.output_dim: int = kwargs.pop("output_dim", 5000)

        self.vocab_size: int = kwargs.pop("vocab_size", 5000)
        self.embedding_dim: int = kwargs.pop("embedding_dim", 256)
        self.input_dim: int = kwargs.pop("input_dim", 5000)
        self.hidden_dim = kwargs.pop("hidden_dim", 512)
        self.num_layers = kwargs.pop("num_layers", 2)
        self.dropout_rate = kwargs.pop("dropout", 0.3)

        self.embedding = nn.Embedding(self.vocab_size, self.input_dim)

        layers: List[nn.Module] = []
        dims = [self.input_dim] + self.hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if self.dropout_rate:
                layers.append(nn.Dropout(self.dropout_rate))

        layers.append(nn.Linear(dims[-1], self.output_dim))

        self.layers = nn.ModuleList(layers)

        super().__init__(model_path=model_path, **kwargs)
        BaseGenerativeModel.__init__(self, model_path=model_path, **kwargs)

    def forward(self, x) -> Tensor:

        embedded = self.embedding(x)
        batch_size, seq_len, embed_dim = embedded.size()
        x = embedded.view(-1, embed_dim)

        for layer in self.layers:
            x = layer(x)

        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, vocab_size)

        return x

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

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):

            input_seq = batch["decoder_input"].to(self.device)
            target_seq = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            output = self.forward(input_seq)

            output = output.reshape(-1, self.vocab_size)
            target_seq = target_seq.reshape(-1)

            # Create a mask to ignore padding tokens in loss calculation
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
        inputs: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> List[str]:

        self.eval()
        if isinstance(inputs, str):
            inputs = [inputs]

        input_token_lists = []
        for prompt in inputs:
            prompt_tokens = self.tokenizer.encode(prompt)
            input_tokens = [self.tokenizer.token_to_id("[SOS]")] + prompt_tokens
            input_token_lists.append(input_tokens)

        max_prompt_len = max(len(toks) for toks in input_token_lists)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        input_token_lists = [
            toks + [pad_id] * (max_prompt_len - len(toks)) for toks in input_token_lists
        ]
        input_tensor = torch.tensor(input_token_lists, dtype=torch.long).to(self.device)

        generated_texts = []
        min_length = 30  # At least 30 tokens before EOS

        with torch.no_grad():

            batch_size = input_tensor.size(0)
            output = self.forward(input_tensor)
            generated_tokens = [[] for _ in range(batch_size)]
            current_tokens = input_tensor[:, -1].unsqueeze(1)
            finished = [False] * batch_size

            for i in range(max_length):

                output, _ = self.forward(current_tokens)
                logits = output[:, -1, :] / temperature  # (batch, vocab)

                for b in range(batch_size):
                    if finished[b]:
                        continue
                    next_token_logits = logits[b]

                    # Block EOS and PAD if min_length not reached
                    if i < min_length:
                        eos_id = self.tokenizer.token_to_id("[EOS]")
                        pad_id = self.tokenizer.token_to_id("[PAD]")
                        next_token_logits[eos_id] = float("-inf")
                        next_token_logits[pad_id] = float("-inf")

                    # Top-k
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(
                            next_token_logits, top_k
                        )
                        mask = torch.full_like(next_token_logits, float("-inf"))
                        mask[top_k_indices] = top_k_logits
                        next_token_logits = mask

                    # Top-p
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

                    # Stop if EOS or PAD and min_length reached
                    if i >= min_length and next_token in [
                        self.tokenizer.token_to_id("[EOS]"),
                        self.tokenizer.token_to_id("[PAD]"),
                    ]:
                        finished[b] = True
                        continue
                    generated_tokens[b].append(next_token)
                    current_tokens[b, 0] = next_token
                if all(finished):
                    break

        for toks in generated_tokens:
            generated_texts.append(self.tokenizer.decode(toks))

        # Always return a list, even for single input
        return generated_texts

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
