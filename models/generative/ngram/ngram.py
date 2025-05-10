from typing import Any
from data.tokenizers.base import BaseTokenizer
from datasets import load_dataset
import pandas as pd
import os
import json
import random

from models.generative.base import BaseGenerativeModel
from .config import ORDER
import nltk
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import math
from data.tokenizers.bpe import BpeTokenizer
from nltk.util import ngrams

nltk.download('punkt', quiet=True)

class NgramGenerator(BaseGenerativeModel):
    def __init__(self, model_path=None, **kwargs):
        self.n = kwargs.pop("order", ORDER)
        self.ngram_probs = {order: defaultdict(float) for order in range(2, self.n + 1)}
        self.unigram_counts = Counter()
        self.vocab = set()
        self.V = 0
        self.tokenizer: BaseTokenizer = kwargs.pop("tokenizer", None)

    def fit(self, X_train, y_train, X_val, y_val):
        print("[NgramGenerator] Starting training...")
        tokenized = [self.tokenizer.encode(s.lower()) for s in X_train]
        print(f"[NgramGenerator] Tokenized {len(tokenized)} sentences.")
        self.unigram_counts = Counter([w for sent in tokenized for w in sent])
        self.V = len(self.unigram_counts)
        self.vocab = set(self.unigram_counts.keys())
        print(f"[NgramGenerator] Vocabulary size: {self.V}")

        # For each n-gram order, compute probabilities
        for order in range(2, self.n + 1):
            n_grams = [list(ngrams(sent, order)) for sent in tokenized]
            n_grams = [gram for sublist in n_grams for gram in sublist]
            ngram_counts = Counter(n_grams)
            # Compute context counts for denominator
            context_counts = Counter()
            for gram, count in ngram_counts.items():
                context = gram[:-1]
                context_counts[context] += count
            for gram, count in ngram_counts.items():
                context = gram[:-1]
                context_count = context_counts[context]
                self.ngram_probs[order][gram] = (count + 1) / (context_count + self.V)
            print(f"[NgramGenerator] Extracted {len(ngram_counts)} unique {order}-grams.")
        print("[NgramGenerator] Training complete.")

    def infer_next_token(self, tokens):
        # Try to infer the next token using n-grams of decreasing size
        for n in range(self.n, 1, -1):
            if len(tokens) >= n - 1:
                context = tuple(tokens[-(n-1):])
                candidates = [(gram[-1], prob) for gram, prob in self.ngram_probs[n].items() if gram[:-1] == context]
                if candidates:
                    next_token = max(candidates, key=lambda x: x[1])[0]
                    return next_token
        # Fallback: pick most common unigram
        if self.unigram_counts:
            fallback_token = self.unigram_counts.most_common(1)[0][0]
            return fallback_token
        return None

    def generate(self, prompt, max_length=50):
        tokens = self.tokenizer.encode(prompt.lower())
        
        # Remove any end-of-string or special tokens at the end of the prompt
        while tokens:
            decoded = self.tokenizer.decode([tokens[-1]])
            if decoded == '':
                tokens.pop()
            else:
                break
            
        out = tokens[:]
        repeat_count = 0
        last_token = None
        for _ in range(max_length):
            next_token = self.infer_next_token(out)
            if not next_token:
                break
            out.append(next_token)
            if next_token == last_token:
                repeat_count += 1
                if repeat_count > 5:
                    break
            else:
                repeat_count = 0
            last_token = next_token
        generated_text = self.tokenizer.decode(out)
        return generated_text

    def perplexity(self, sentences):
        tokenized = [self.tokenizer.encode(s.lower()) for s in sentences]
        N = sum(len(sent) for sent in tokenized)
        log_prob = 0
        for sent in tokenized:
            # Skip empty sentences
            if len(sent) == 0:
                continue
            for i in range(len(sent)):
                found = False
                # Try to use the highest order n-gram available
                for n in range(self.n, 1, -1):
                    if i - n + 1 >= 0:
                        gram = tuple(sent[i - n + 1:i + 1])
                        prob = self.ngram_probs[n].get(gram, None)
                        if prob is not None:
                            log_prob += -math.log(prob)
                            found = True
                            break
                if not found:
                    # Fallback to unigram probability
                    unigram = sent[i]
                    prob = (self.unigram_counts.get(unigram, 0) + 1) / (sum(self.unigram_counts.values()) + self.V)
                    log_prob += -math.log(prob)
        return math.exp(log_prob / N) 

    def evaluate(
        self,
        X,
        y=None,
        y_pred=None,
    ):
        """
        Evaluate the model on the given data using perplexity.
        Also, for several representative prompts, generate text and return the results.
        Returns a dictionary with the perplexity score and generated samples.
        """
        ppl = self.perplexity(X)
        
        prompts = [
            "The hotel was",
            "Our stay at the resort",
            "The room was clean and",
            "Breakfast was included and",
            "I would not recommend",
        ]

        generations = []
        for prompt in prompts:
            generated = self.generate(prompt, max_length=20)
            generations.append({"prompt": prompt, "generated": generated})

        return {"perplexity": ppl, "generated_samples": generations}

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'n': self.n, 'ngram_probs': {order: {str(k): v for k, v in probs.items()} for order, probs in self.ngram_probs.items()},
                       'vocab': list(self.vocab)}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        obj = cls(model_path=path, order=ORDER, tokenizer=BpeTokenizer())
        obj.ngram_probs = {int(order): {eval(k): v for k, v in probs.items()} for order, probs in data['ngram_probs'].items()}
        obj.vocab = set(data['vocab'])
        obj.V = len(obj.vocab)
        return obj

def main():
    checkpoint_dir = ''
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, 'model.json')
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')

    dataset = load_dataset("jniimi/tripadvisor-review-rating")
    raw_data = pd.DataFrame(dataset['train'])
    df = raw_data.drop(columns=['stay_year', 'post_date', 'freq', 'lang'])
    df = df.dropna().drop_duplicates()
    df = df.sample(n=min(10000, len(df))).reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_texts = list(train_df['review'])
    test_texts = list(test_df['review'])

    # Train or load tokenizer
    if os.path.exists(tokenizer_path):
        tokenizer = BpeTokenizer()
        tokenizer.load(tokenizer_path)
        print("Loaded BPE tokenizer.")
    else:
        tokenizer = BpeTokenizer()
        tokenizer.fit(train_texts)
        tokenizer.save(tokenizer_path)
        print(f"Trained and saved BPE tokenizer to {tokenizer_path}")

    if os.path.exists(model_path):
        print("Loading existing ngram model...")
        model = NgramGenerator.load(model_path)
    else:
        print("Training ngram model...")
        model = NgramGenerator()
        model.fit(train_texts, None, None, None)
        model.save(model_path)
        print(f"Model saved to {model_path}")

    # Generation example
    prompt = "The hotel was"
    print("\nPrompt:", prompt)
    print("Generated:", model.generate(prompt, max_length=40))

    # Perplexity evaluation
    ppl = model.perplexity(test_texts[:100])
    print(f"\nPerplexity on test set: {ppl:.2f}")

if __name__ == "__main__":
    main()
