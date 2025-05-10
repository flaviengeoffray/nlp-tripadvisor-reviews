#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json

from utils import load_config, load_tokenizer
from data.dataprep import prepare_data
from models.generative.rnn.rnn import RNNGenModel

def fix_spacing(text):
    """
    Add spaces to text based on common patterns and capitalization
    """
    import re
    
    # First clean up existing spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Add space before capital letters that aren't at the beginning of words
    result = ""
    for i, char in enumerate(text):
        if i > 0 and char.isupper() and text[i-1] not in [' ', '-', '(', '[', '{', '.', '!', '?', ':', ';']:
            result += " " + char
        else:
            result += char
    
    # Apply regex replacements for common punctuation patterns
    spacing_patterns = [
        (r'\.(?=[A-Z])', '. '),     # Period followed by capital letter
        (r',(?=[^\s])', ', '),      # Comma not followed by space
        (r'!(?=[^\s])', '! '),      # Exclamation mark not followed by space
        (r'\?(?=[^\s])', '? '),     # Question mark not followed by space
        (r':(?=[^\s])', ': '),      # Colon not followed by space
        (r';(?=[^\s])', '; '),      # Semicolon not followed by space
        (r'(?<=[^\s])-(?=[^\s])', ' - '),  # Hyphen with no spaces
        (r'\'s(?=[^\s])', '\'s '),  # Possessive not followed by space
    ]
    
    for pattern, replacement in spacing_patterns:
        result = re.sub(pattern, replacement, result)
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result)
    
    # Trim leading/trailing whitespace
    result = result.strip()
    
    return result

def test_model(model, tokenizer, test_data, num_samples=10):
    """
    Test the model by generating reviews and comparing with original reviews
    """
    model.eval()
    results = []
    
    # Select random samples from test data
    indices = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)
    selected_samples = test_data.iloc[indices]
    
    for _, row in tqdm(selected_samples.iterrows(), total=len(selected_samples), desc="Testing"):
        original_review = row['review']
        rating = int(row['overall'])  # Assuming rating is 1-5
        
        # Extract keywords from the original review (simple approach)
        words = original_review.lower().split()
        # Filter common words, keep only longer words as potential keywords
        keywords = [word for word in words if len(word) > 5 and word.isalpha()]
        keywords = list(set(keywords))[:5]  # Remove duplicates and keep top 5
        keywords_str = ", ".join(keywords) if keywords else "hotel, stay"
        
        try:
            # Generate new review
            generated_review = model.generate(
                rating=rating,
                keywords=keywords_str,
                max_length=200,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            # Apply fix to add spaces to the generated text
            generated_review = fix_spacing(generated_review)
            
        except Exception as e:
            print(f"Error generating review: {e}")
            generated_review = "Error: " + str(e)
        
        # Test tokenizer decoding/encoding
        sample_text = original_review[:100] if len(original_review) > 100 else original_review
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        
        # Apply fix to decoded text as well
        fixed_decoded = fix_spacing(decoded)
        
        results.append({
            "rating": rating,
            "keywords": keywords_str,
            "original_review_sample": sample_text,
            "generated_review": generated_review,
            "tokenizer_test": {
                "original": sample_text,
                "decoded": decoded,
                "fixed_decoded": fixed_decoded,
                "match": sample_text == decoded
            }
        })
        
    return results

def diagnose_generation_issues(model, tokenizer):
    """
    Diagnose issues with the model generation process
    """
    print("Diagnosing model generation issues...")
    
    # Test 1: Simple encoding-decoding cycle
    test_texts = [
        "This hotel was amazing!",
        "The service was terrible.",
        "We had a wonderful time at the beach resort."
    ]
    
    print("\nTest 1: Tokenizer encoding-decoding")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: '{text}'")
        print(f"Tokens: {encoded}")
        print(f"Decoded: '{decoded}'")
        print(f"Match: {text == decoded}")
        print("---")
    
    # Test 2: Simple generation with controlled input
    print("\nTest 2: Simple generation test")
    ratings = [1, 3, 5]
    keywords = "hotel, beach, service"
    
    for rating in ratings:
        print(f"\nGenerating review for rating {rating} with keywords: '{keywords}'")
        try:
            with torch.no_grad():
                generated = model.generate(
                    rating=rating,
                    keywords=keywords,
                    max_length=50,  # Short for testing
                    temperature=1.0,
                    top_k=0,  # Disable top-k to see raw model output
                    top_p=0
                )
            print(f"Generated: '{generated}'")
            
            # Analyze token probabilities for a short prompt
            prompt = f"{rating}: {keywords}"
            prompt_tokens = [tokenizer.token_to_id("[SOS]")] + tokenizer.encode(prompt)
            input_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(model.device)
            
            output, _ = model(input_tensor)
            next_token_logits = output[0, -1, :]
            top_logits, top_indices = torch.topk(next_token_logits, 10)
            top_probs = torch.softmax(top_logits, dim=-1)
            
            print("Top 10 next tokens:")
            for i, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist())):
                token = tokenizer.id_to_token(idx)
                print(f"{i+1}. '{token}' (ID: {idx}, Prob: {prob:.4f})")
                
        except Exception as e:
            print(f"Error during generation: {str(e)}")
    
    # Test 3: Check special tokens
    print("\nTest 3: Special token test")
    special_tokens = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"Token: {token}, ID: {token_id}")
    
    # Output device and model info
    print(f"\nModel device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Vocab size: {model.vocab_size}")

def main(config_path: str, checkpoint_path: str, output_path: str = None, num_samples: int = 10):
    """
    Main function to test a trained RNN model
    """
    config = load_config(config_path)
    
    # Load test data
    print("Loading test data...")
    _, _, test_df = prepare_data(
        dataset_name=config.dataset_name,
        label_col=config.label_col,
        drop_columns=["stay_year", "post_date", "freq", "lang"],
        test_size=config.test_size,
        val_size=config.val_size,
        seed=config.seed,
        stratify=config.stratify,
        sample_size=config.sample_size,
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
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
    print("Loading model...")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")
    
    state = torch.load(checkpoint, map_location='cpu')
    
    # Create model with parameters from checkpoint
    model = RNNGenModel(
        model_path=checkpoint.parent,
        vocab_size=state.get("vocab_size", 5000),
        embedding_dim=state.get("embedding_dim", 256),
        hidden_dim=state.get("hidden_dim", 512),
        num_layers=state.get("num_layers", 2),
        dropout=state.get("dropout", 0.3),
        tokenizer=tokenizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Load weights
    model.load(checkpoint)
    print(f"Model loaded successfully from {checkpoint}")
    
    # Run diagnostics
    diagnose_generation_issues(model, tokenizer)
    
    # Test model
    print(f"\nTesting model on {num_samples} random samples...")
    results = test_model(model, tokenizer, test_df, num_samples=num_samples)
    
    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Test results saved to {output_file}")
    
    # Print summary
    successful_generations = sum(1 for r in results if r["generated_review"] and len(r["generated_review"]) > 10)
    print(f"\nSummary: {successful_generations}/{len(results)} successful generations")
    
    tokenizer_matches = sum(1 for r in results if r["tokenizer_test"]["match"])
    print(f"Tokenizer encoding/decoding matches: {tokenizer_matches}/{len(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained RNN model")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--checkpoint", "-m", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output", "-o", type=str, default="test_results.json", help="Path to save test results")
    parser.add_argument("--samples", "-n", type=int, default=10, help="Number of test samples")
    
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output, args.samples)
