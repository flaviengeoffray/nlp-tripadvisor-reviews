#!/usr/bin/env python3
# generate.py

import argparse
from pathlib import Path

import torch

from utils import load_config, load_vectorizer, load_model

def main():
    parser = argparse.ArgumentParser(
        description="Generate a TripAdvisor review with your feed-forward model"
    )
    parser.add_argument(
        "--config","-c", required=True, help="Path to your YAML config"
    )
    parser.add_argument(
        "--rating","-r", type=int, choices=range(1,6), required=True,
        help="Overall rating (1–5) to condition generation on",
    )
    parser.add_argument(
        "--keywords","-k", default="", help="Comma-separated keywords"
    )
    parser.add_argument(
        "--device","-d", choices=["cpu","cuda"], default=None,
        help="Device to run on (defaults to CUDA if available)",
    )
    args = parser.parse_args()

    # 1) Load config
    config = load_config(args.config)

    # 2) Instantiate & load your TF-IDF wrapper
    vectorizer = load_vectorizer(config.vectorizer)
    if config.vectorizer.checkpoint:
        vectorizer.load(Path(config.vectorizer.checkpoint))
    else:
        vectorizer.load(Path(config.model_path) / "vectorizer.bz2")

    # 3) Pull out the underlying sklearn vectorizer
    sk_vec = vectorizer.vectorizer

    # 4) Get feature names from sklearn object
    if hasattr(sk_vec, "get_feature_names_out"):
        feat_names = sk_vec.get_feature_names_out()
    elif hasattr(sk_vec, "get_feature_names"):
        feat_names = sk_vec.get_feature_names()
    else:
        vocab = sk_vec.vocabulary_
        feat_names = sorted(vocab, key=vocab.get)

    true_vocab_size = len(feat_names)

    # 5) Patch model dims so its final linear matches this vocab size
    params = config.model.params or {}
    params["input_dim"]  = true_vocab_size
    params["vocab_size"] = true_vocab_size
    config.model.params = params

    # 6) Load model and its weights
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(config.model, Path(config.model_path))
    if config.model.checkpoint:
        model.load(config.model.checkpoint)

    # 7) Attach vectorizer, move to device, set eval
    model.vectorizer = vectorizer
    model.to(device)
    model.eval()

    # 8) Build the prompt and generate
    prompt = f"{args.rating}: {args.keywords}".strip()
    print(f"\nPrompt → {prompt}\n")
    review = model.generate(prompt)
    print("Generated review:\n")
    print(review)
    print()

if __name__ == "__main__":
    main()
