#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
import yaml
from typing import Dict, Any
import json

from utils import load_config, load_tokenizer
from models.generative.rnn.rnn import RNNGenModel

def load_model_from_checkpoint(checkpoint_path: Path, model_config: Dict[str, Any], tokenizer) -> RNNGenModel:
    """
    Load a trained RNN model from a checkpoint
    """
    # Extract model parameters from config
    vocab_size = model_config.get("vocab_size", 5000)
    embedding_dim = model_config.get("embedding_dim", 256)
    hidden_dim = model_config.get("hidden_dim", 512)
    num_layers = model_config.get("num_layers", 2)
    dropout = model_config.get("dropout", 0.3)
    
    # Create the model with the same architecture
    model = RNNGenModel(
        model_path=checkpoint_path.parent,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        tokenizer=tokenizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Load the model weights
    model.load(checkpoint_path)
    return model

def main(config_path: str, checkpoint_path: str, output_path: str = None):
    """
    Main function to generate reviews using a trained RNN model
    """
    config = load_config(config_path)
    
    # Load tokenizer
    if config.tokenizer:
        tokenizer = load_tokenizer(config.tokenizer)
        tokenizer_path = config.model_path / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer.load(str(tokenizer_path))
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    else:
        raise ValueError("Tokenizer configuration is required")
    
    # Load model
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")
    
    model = load_model_from_checkpoint(checkpoint, config.model.params or {}, tokenizer)
    print(f"Model loaded successfully from {checkpoint}")
    
    # Generate reviews for different ratings and keywords
    ratings = [1, 2, 3, 4, 5]
    keywords_list = [
        "beach, ocean, resort",
        "food, restaurant, service",
        "room, cleanliness, comfort",
        "staff, friendly, helpful",
        "price, value, expensive"
    ]
    
    results = []
    
    for rating in ratings:
        for keywords in keywords_list:
            print(f"\nGenerating review for rating {rating} with keywords: {keywords}")
            
            # Generate with different temperatures
            # Temperature controls the randomness of the output
            # Lower temperature = more deterministic output
            # Higher temperature = more random output
            for temp in [1.0]:
                generated_text = model.generate(
                    rating=rating,
                    keywords=keywords,
                    max_length=200,
                    temperature=temp,
                    top_k=50,
                    top_p=0.9
                )
                                
                print(f"\nTemperature {temp}:")
                print(f"Generated (raw): {generated_text}")
                
                results.append({
                    "rating": rating,
                    "keywords": keywords,
                    "temperature": temp,
                    "generated_text": generated_text,
                })
    
    # Save results to file if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nGenerated reviews saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reviews using a trained RNN model")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--checkpoint", "-m", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output", "-o", type=str, default="generated_reviews.json", help="Path to save generated reviews")
    
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output)
