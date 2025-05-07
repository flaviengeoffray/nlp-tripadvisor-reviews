from pathlib import Path
from typing import Any, Optional, Sequence, Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

import numpy as np

from models.base_pytorch import BaseTorchModel
from models.generative.base import BaseGenerativeModel


class FNNGenerativeModel(BaseTorchModel, BaseGenerativeModel):
    """
    A feedforward neural network generative model for next-token prediction.
    """

    def __init__(self, model_path: Path, **kwargs: Any):
        nn.Module.__init__(self)

        # Network hyperparameters (configurable)
        input_dim: int = kwargs.pop("input_dim", 1000)
        hidden_dims: Sequence[int] = kwargs.pop("hidden_dims", [256, 256])
        vocab_size: int = kwargs.pop("vocab_size", input_dim)
        dropout_rate: float = kwargs.pop("dropout_rate", 0.3)
        lr: float = kwargs.pop("lr", 1e-3)
        self.max_length: int = kwargs.pop("max_length", 50)
        self.temperature: float = kwargs.pop("temperature", 1.0)

        # Build MLP layers
        layers: List[nn.Module] = []
        dims = [input_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(dims[-1], vocab_size))
        self.layers = nn.ModuleList(layers)

        # Initialize BaseTorchModel (handles optimizer, scheduler, etc.)
        super().__init__(model_path=model_path, **kwargs)
        # Auto-load TF-IDF vectorizer if saved alongside the model
        try:
            from vectorizers.tfidf import TfidfVectorizer
            vpath = model_path / "vectorizer.bz2"
            if vpath.exists():
                tfidf = TfidfVectorizer()
                tfidf.load(vpath)
                self.vectorizer = tfidf
        except Exception:
            # If loading fails, leave vectorizer unset
            pass

    def forward(self, X: Tensor) -> Tensor:
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def _get_feature_names(self) -> List[str]:
        # Access the underlying sklearn vectorizer if wrapped
        vec = self.vectorizer
        # Direct methods
        if hasattr(vec, "get_feature_names_out"):
            return list(vec.get_feature_names_out())
        if hasattr(vec, "get_feature_names"):
            return vec.get_feature_names()
        # Wrapped sklearn vect
        sk = getattr(vec, "vectorizer", None)
        if sk is not None:
            if hasattr(sk, "get_feature_names_out"):
                return list(sk.get_feature_names_out())
            if hasattr(sk, "get_feature_names"):
                return sk.get_feature_names()
            vocab = getattr(sk, "vocabulary_", None)
            if vocab:
                return sorted(vocab, key=vocab.get)
        # Fallback numeric strings
        return []

    def generate(self, input_data: Optional[str] = None) -> str:
        """
        Generate a sequence of tokens conditioned on the input prompt.
        """
        if not hasattr(self, "vectorizer"):
            raise AttributeError("Attach a TF-IDF vectorizer to the model before generating.")

        feature_names = self._get_feature_names()
        if not feature_names:
            raise RuntimeError("Unable to retrieve feature names from the vectorizer.")

        prompt = input_data or ""
        current_text = prompt.strip()
        generated_tokens: List[str] = []

        self.eval()
        with torch.no_grad():
            for _ in range(self.max_length):
                X_tfidf = self.vectorizer.transform([current_text])
                X_tensor = torch.tensor(X_tfidf.toarray(), dtype=torch.float32).to(self.device)

                logits = self.forward(X_tensor)[0]
                probs = torch.softmax(logits / self.temperature, dim=0)
                idx = torch.multinomial(probs, num_samples=1).item()

                generated_tokens.append(feature_names[idx])
                current_text += " " + feature_names[idx]

        return " ".join(generated_tokens)

    def _train_loop(self, train_loader: DataLoader, epoch: int) -> float:
        total_loss = 0.0
        self.train()
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.forward(X)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _val_loop(self, val_loader: DataLoader) -> Tuple[List[np.ndarray], List[int], float]:
        total_loss = 0.0
        all_preds: List[np.ndarray] = []
        all_labels: List[int] = []
        self.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.forward(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy().tolist())
        return all_preds, all_labels, total_loss

    def evaluate(
        self,
        X: Any = None,
        y: Any = None,
        y_pred: Any = None
    ) -> Dict[str, float]:
        """
        Evaluate the model:
        - If a vectorizer is attached, flatten and map numeric indices to token strings,
          then compute CER, WER, and BLEU.
        - Otherwise, delegate to the base class evaluation directly.
        """
        # If no vectorizer, skip token mapping and use default evaluation
        if not hasattr(self, "vectorizer") or self.vectorizer is None:
            return super().evaluate(X, y, y_pred)

        # Fetch feature names for mapping
        feature_names = self._get_feature_names()
        if not feature_names:
            # fallback if mapping fails
            return super().evaluate(X, y, y_pred)

        # Flatten batch outputs
        if isinstance(y_pred, list):
            flat_preds = np.concatenate(y_pred, axis=0)
        else:
            flat_preds = np.asarray(y_pred) if y_pred is not None else np.array([])
        flat_labels = np.asarray(y) if y is not None else np.array([])

        # Map numeric indices to token strings
        pred_tokens = [feature_names[int(idx)] for idx in flat_preds]
        true_tokens = [feature_names[int(idx)] for idx in flat_labels]

        # Compute and return metrics casting to floats
        raw_metrics = super().evaluate(X=None, y=true_tokens, y_pred=pred_tokens)
        return {key: float(val) for key, val in raw_metrics.items()}
