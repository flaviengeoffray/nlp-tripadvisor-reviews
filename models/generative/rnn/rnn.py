import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from data.datasets.TripAdvisorDataset import TripAdvisorDataset, causal_mask

from data.tokenizers.base import BaseTokenizer
from data.tokenizers.bpe import BpeTokenizer
from models.base_pytorch import BaseTorchModel
from models.generative.base import BaseGenerativeModel

class RNNGenModel(BaseTorchModel, BaseGenerativeModel):
    def __init__(
        self,
        model_path,
        **kwargs
    ):
        nn.Module.__init__(self)
        self.vocab_size: int = kwargs.pop("vocab_size", 5000)  
        self.embedding_dim: int = kwargs.pop("embedding_dim", 256)
        self.hidden_dim = kwargs.pop("hidden_dim", 512)
        self.num_layers = kwargs.pop("num_layers", 2)
        self.dropout_rate = kwargs.pop("dropout", 0.3)

        # Define the model architecture
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)

        # Initialize base classes properly
        BaseTorchModel.__init__(self, model_path=model_path, **kwargs)
        BaseGenerativeModel.__init__(self, model_path)

    def forward(self, x, hidden=None):
        """
        Forward pass through the LSTM Generator
        
        Args:
            x: Input tensor of token IDs [batch_size, seq_len]
            hidden: Initial hidden state (optional)
            
        Returns:
            output: Probability distribution over vocabulary for next token
            hidden: Updated hidden state
        """
        # x shape: [batch_size, seq_len]
        batch_size = x.size(0)
        
        # Embed the input
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
            hidden = (h0, c0)
        
        # Pass through LSTM
        output, hidden = self.lstm(embedded, hidden)
        # output shape: [batch_size, seq_len, hidden_dim]
        
        # Apply dropout
        output = self.dropout(output)
        
        # Pass through fully connected layer
        output = self.fc(output)
        # output shape: [batch_size, seq_len, vocab_size]
        
        return output, hidden
    
    def _get_dataloaders(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        shuffle: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Override the base _get_dataloaders method to handle text data properly
        """
        # For generative models, we'll use a custom dataset
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
            
            # Stack tensors
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
            # Get input and target sequences
            input_seq = batch["decoder_input"].to(self.device)
            target_seq = batch["label"].to(self.device)
            mask = batch["decoder_mask"].squeeze(1).to(self.device)  # Remove batch dimension from mask
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output, _ = self.forward(input_seq)
            
            # Reshape output and target for cross entropy loss
            # output: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
            # target: [batch_size, seq_len] -> [batch_size * seq_len]
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
    
    
    def evaluate(self, X=None, y=None, y_pred=None):
        """Evaluate the model with NLP metrics"""
        from torchmetrics.text import BLEUScore, WordErrorRate, ROUGEScore
        
        # Default metrics for invalid inputs
        default_metrics = {"bleu": 0.0, "rouge-1": 0.0, "wer": 1.0, "diversity": 0.0}
        
        # Validate inputs
        if y is None or y_pred is None or len(y) == 0:
            print("WARNING: Missing evaluation inputs")
            return default_metrics
            
        # Process predictions
        decoded_preds = []
        pad_id = self.tokenizer.token_to_id("[PAD]")
        sos_id = self.tokenizer.token_to_id("[SOS]")
        eos_id = self.tokenizer.token_to_id("[EOS]")
        
        # Extract valid predictions
        for batch in y_pred:
            if not isinstance(batch, np.ndarray):
                continue
                
            for pred_seq in batch:
                # Make sure pred_seq is an array
                if not hasattr(pred_seq, '__iter__'):
                    continue
                    
                # Truncate at EOS if present
                eos_indices = np.where(np.array(pred_seq) == eos_id)[0]
                if len(eos_indices) > 0:
                    pred_seq = pred_seq[:eos_indices[0]]
                
                # Remove special tokens
                valid_tokens = [t for t in pred_seq if t != pad_id and t != sos_id and t != eos_id]
                
                if valid_tokens:
                    decoded_preds.append(self.tokenizer.decode(valid_tokens))
        
        if not decoded_preds:
            return default_metrics
        
        # Prepare data for metrics calculation
        n = min(len(decoded_preds), len(y))
        candidates = decoded_preds[:n]
        references = [[text] for text in y[:n]]  # BLEU format
        ref_texts = [ref[0] for ref in references]  # Flat references
        
        # Calculate metrics
        bleu_score = BLEUScore(n_gram=4)(candidates, references).item()
        rouge_results = ROUGEScore()(candidates, ref_texts)
        wer_score = WordErrorRate()(candidates, ref_texts).item()
        
        # Calculate lexical diversity
        words = ' '.join(candidates).lower().split()
        diversity = len(set(words)) / max(len(words), 1)
        
        # Compile and save metrics
        metrics = {
            "bleu": bleu_score,
            "rouge-1": rouge_results["rouge1_fmeasure"].item(),
            "rouge-l": rouge_results["rougeL_fmeasure"].item(),
            "wer": wer_score,
            "diversity": diversity
        }
        
        # Save to file
        with open(Path(self.model_path) / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        return metrics

    
    def generate(
        self, 
        rating: int, 
        keywords: str, 
        max_length: int = 200, 
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a TripAdvisor review based on rating and keywords
        
        Args:
            rating: Rating from 1-5
            keywords: Keywords to use in generation
            max_length: Maximum length of generated text
            temperature: Temperature for sampling (higher = more random)
            top_k: If > 0, sample from top k most likely tokens
            top_p: If > 0, sample from tokens with cumulative probability > p
            
        Returns:
            generated_text: The generated review text
        """
        self.eval()
        
        # Create prompt: "rating: keywords"
        prompt = f"{rating}: {keywords}"
        
        # Tokenize the prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        
        # Add SOS token at the beginning
        input_tokens = [self.tokenizer.token_to_id("[SOS]")] + prompt_tokens
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
        
        generated_tokens = []
        
        # Initialize hidden state
        hidden = None
        
        with torch.no_grad():
            # First forward pass with the prompt
            output, hidden = self.forward(input_tensor, hidden)
            
            # Now generate tokens one by one
            current_token = input_tensor[:, -1].unsqueeze(1)  # Use last token as input
            
            # Generate up to max_length tokens
            for _ in range(max_length):
                # Get model prediction for next token
                output, hidden = self.forward(current_token, hidden)
                
                # Get logits for the next token prediction
                next_token_logits = output[0, -1, :] / temperature
                
                # Apply top-k sampling if specified
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) sampling if specified
                if top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create mask for indices to remove
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # If we hit EOS token or PAD token, stop generation
                if next_token == self.tokenizer.token_to_id("[EOS]") or next_token == self.tokenizer.token_to_id("[PAD]"):
                    break
                
                # Add token to generated sequence
                generated_tokens.append(next_token)
                
                # Update input tensor for next iteration
                current_token = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
        
        # Decode the generated tokens
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text
    
    def save(self, path: Path, epoch: int = None) -> None:
        """
        Save the model
        
        Args:
            path: Path to save model
            epoch: Current epoch number (optional)
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
        
        Args:
            path: Path to load model from
        """
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.start_epoch = state.get("epoch", 0) + 1
        
        # Optionally update model parameters if they're in the saved state
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
