import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

import torchmetrics
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.rouge import ROUGEScore

from data.datasets.TripAdvisorDataset import TripAdvisorDataset, causal_mask

from data.tokenizers.base import BaseTokenizer
from data.tokenizers.bpe import BpeTokenizer
from models.base_pytorch import BaseTorchModel
from models.generative.base import BaseGenerativeModel


class FNNGenerativeModel(BaseTorchModel, BaseGenerativeModel):

    def __init__(self, model_path: Path, **kwargs: Any):        
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

    def forward(self, x, hidden=None):
        """
        Forward pass through the Feedforward Network
        
        Args:
            x: Input tensor of token IDs [batch_size, seq_len]
            hidden: Not used in feedforward networks, included for API compatibility
        
        Returns:
            output: Probability distribution over output classes
            hidden: None (for API compatibility with RNN-type models)
        """
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x) 
        batch_size, seq_len, embed_dim = embedded.size()
        x = embedded.view(-1, embed_dim)

        for layer in self.layers:
            x = layer(x)

        x = x.view(batch_size, seq_len, -1)  # Reshape back to [batch_size, seq_len, vocab_size]

        return x, None
    
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
            # Create a dictionary to store batch data
            batch_data = {
                "encoder_input": [],
                "decoder_input": [],
                "encoder_mask": [],
                "decoder_mask": [],
                "label": [],
                "source_text": [],
                "target_text": []
            }
            
            # Collect all items across the batch
            for item in batch:
                for key in batch_data:
                    batch_data[key].append(item[key])
            
            for key in ["encoder_input", "decoder_input", "encoder_mask", "decoder_mask", "label"]:
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
        """
        Training loop for one epoch
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            train_loss: Average training loss for this epoch
        """
        self.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):

            input_seq = batch["decoder_input"].to(self.device)
            target_seq = batch["label"].to(self.device)
            mask = batch["decoder_mask"].squeeze(1).to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output, _ = self.forward(input_seq)
            
            output = output.reshape(-1, self.vocab_size)
            target_seq = target_seq.reshape(-1)
            
            # Create a mask to ignore padding tokens in loss calculation
            # -1 for ignored positions (PAD tokens)
            pad_mask = target_seq != self.tokenizer.token_to_id("[PAD]")
            
            # Create masked targets and outputs
            masked_target = target_seq[pad_mask]
            masked_output = output[pad_mask]
            
            # Calculate loss
            loss = self.criterion(masked_output, masked_target)
            
            # Backward pass and optimize
            loss.backward()
            
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            train_loss += loss.item()
            
        return train_loss
 
    def _val_loop(self, val_loader: DataLoader) -> Tuple[List[np.ndarray], List[str], float]:
        """
        Validation loop
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            all_preds: List of prediction arrays
            all_labels: List of true labels
            val_loss: Total validation loss
        """
        self.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get input and target sequences
                input_seq = batch["decoder_input"].to(self.device)
                target_seq = batch["label"].to(self.device)
                
                # Forward pass
                output, _ = self.forward(input_seq)
                
                # Get predictions
                _, preds = torch.max(output, dim=2)
                
                # Calculate loss
                output_flat = output.reshape(-1, self.vocab_size)
                target_flat = target_seq.reshape(-1)
                pad_mask = target_flat != self.tokenizer.token_to_id("[PAD]")
                loss = self.criterion(output_flat[pad_mask], target_flat[pad_mask])
                val_loss += loss.item()
                
                # Store predictions
                all_preds.append(preds.cpu().numpy())
                
                # Convert targets to text
                for i in range(target_seq.size(0)):
                    seq = target_seq[i].cpu().numpy()
                    eos_pos = np.where(seq == self.tokenizer.token_to_id("[EOS]"))[0]
                    if len(eos_pos) > 0:
                        seq = seq[:eos_pos[0]+1]
                    
                    # Filter out special tokens
                    seq = [t for t in seq if t not in [
                        self.tokenizer.token_to_id("[PAD]"),
                        self.tokenizer.token_to_id("[SOS]")
                    ]]
                    
                    text = self.tokenizer.decode(seq)
                    all_labels.append(text)

        return all_preds, all_labels, val_loss
    
    
    def evaluate(self, X: Union[np.ndarray, Tensor] = None, y: Union[np.ndarray, Tensor] = None, y_pred: Union[np.ndarray, Tensor] = None) -> Dict[str, float]:
        """Evaluate the model with NLP metrics using torchmetrics"""

        if y_pred is None:
            # If predictions are not provided, generate them
            _, y_pred, _ = self._val_loop(self._get_dataloaders(X, y, X, y, shuffle=False)[1])

        # Initialize metrics
        bleu = BLEUScore(n_gram=1, smooth=True)
        rouge = ROUGEScore()
        wer = torchmetrics.WordErrorRate()
        cer = torchmetrics.CharErrorRate()
        
        # Calculate BLEU score
        # Ensure y_pred is a lists of strings
        y_pred = [self.tokenizer.decode(pred) for pred in y_pred]

        bleu_score = bleu(y_pred, y)

        # Calculate ROUGE scores
        rouge_scores = rouge(y_pred, y)

        # Calculate WER and CER
        wer_score = wer(y_pred, y)
        cer_score = cer(y_pred, y)

        return {
            "BLEU": bleu_score.item(),
            "ROUGE-1": rouge_scores["rouge1_fmeasure"].item(),
            "ROUGE-2": rouge_scores["rouge2_fmeasure"].item(),
            "ROUGE-L": rouge_scores["rougeL_fmeasure"].item(),
            "WER": wer_score.item(),
            "CER": cer_score.item(),
        }
            

    
    def generate(self, rating, keywords, max_length=200, temperature=1.0, top_k=50, top_p=0.9):
        """Generate a review based on rating and keywords"""
        self.eval()
        
        # Create prompt
        prompt = f"{rating}: {keywords}"
        prompt_tokens = self.tokenizer.encode(prompt)
        
        # Prepare input with SOS token
        input_tokens = [self.tokenizer.token_to_id("[SOS]")] + prompt_tokens
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(self.device)

        generated_tokens = []
        
        min_length = 30  # At least 30 tokens before EOS
        
        with torch.no_grad():   
            # First pass with the prompt
            output, _ = self.forward(input_tensor)
            current_token = input_tensor[:, -1].unsqueeze(1)  # Last token
            
            # Generate tokens one by one
            for i in range(max_length):
                output, _ = self.forward(current_token)
                next_token_logits = output[0, -1, :] / temperature
                
                # Block EOS and PAD tokens if the minimum length has not been reached
                if i < min_length:
                    eos_id = self.tokenizer.token_to_id("[EOS]")
                    pad_id = self.tokenizer.token_to_id("[PAD]")
                    next_token_logits[eos_id] = float('-inf')
                    next_token_logits[pad_id] = float('-inf')
                
                # Apply top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Apply top-p sampling
                if top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # Stop if EOS or PAD and minimum length is reached
                if i >= min_length and next_token in [
                    self.tokenizer.token_to_id("[EOS]"),
                    self.tokenizer.token_to_id("[PAD]")
                ]:
                    break
                
                generated_tokens.append(next_token)
                current_token = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
        
        # Decode the generated tokens
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text
    
    def save(self, path: Path, epoch: int = None) -> None:
        """
        Save the model
        """
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
        """
        Load the model
        """
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
