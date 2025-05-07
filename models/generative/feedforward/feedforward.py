from pathlib import Path
from typing import Any, Optional, Sequence, Dict

import torch
from torch import Tensor, nn

from models.base_pytorch import BaseTorchModel
from models.generative.base import BaseGenerativeModel


class FNNGenerativeModel(BaseTorchModel, BaseGenerativeModel):
    """
    Feedforward neural network for text generation using TF-IDF features.
    """
    def __init__(self, model_path: Path, **kwargs: Any):
        # Initialize nn.Module
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
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-1], vocab_size))

        self.layers = nn.ModuleList(layers)

        # Initialize BaseTorchModel (handles optimizer, scheduler, etc.)
        super().__init__(model_path=model_path, kwargs=kwargs)

    def forward(self, X: Tensor) -> Tensor:
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def generate(self, input_data: Optional[str] = None) -> str:
        """
        Generate a sequence of tokens conditioned on the input prompt.
        """
        if not hasattr(self, "vectorizer"):
            raise AttributeError("Attach a TF-IDF vectorizer to the model before generating.")

        prompt = input_data or ""
        current_text = prompt.strip()
        generated_tokens = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(self.max_length):
                # Vectorize current text
                X_tfidf = self.vectorizer.transform([current_text])
                X_tensor = torch.tensor(X_tfidf.toarray(), dtype=torch.float32).to(self.device)

                # Compute logits and sample next token
                logits = self.forward(X_tensor)[0]
                probs = torch.softmax(logits / self.temperature, dim=0)
                idx = torch.multinomial(probs, num_samples=1).item()

                token = self.vectorizer.get_feature_names_out()[idx]
                generated_tokens.append(token)
                current_text += " " + token

        return " ".join(generated_tokens)

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Skip evaluation during training for generative model.
        """
        return {}
