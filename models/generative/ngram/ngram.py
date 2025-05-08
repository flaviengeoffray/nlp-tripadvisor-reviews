from data.tokenizers.base import BaseTokenizer
from datasets import load_dataset
import pandas as pd
import os
import json

from models.generative.base import BaseGenerativeModel
from .config import ORDER
import nltk
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import math
from data.tokenizers.bpe import BpeTokenizer

nltk.download('punkt', quiet=True)

class NgramGenerator(BaseGenerativeModel):
    def __init__(self, model_path, **kwargs):
        self.n = kwargs.pop("order", ORDER)
        self.ngram_probs = defaultdict(float)
        self.unigram_counts = Counter()
        self.vocab = set()
        self.V = 0
        self.tokenizer: BaseTokenizer = kwargs.pop("tokenizer", BpeTokenizer())

    def fit(self, X_train, y_train, X_val, y_val):
        tokenized = [self.tokenizer.encode(s.lower()) for s in X_train]
        self.unigram_counts = Counter([w for sent in tokenized for w in sent])
        self.V = len(self.unigram_counts)
        self.vocab = set(self.unigram_counts.keys())
        n_grams = [tuple(gram) for sent in tokenized for gram in nltk.ngrams(sent, self.n)]
        ngram_counts = Counter(n_grams)
        for gram, count in ngram_counts.items():
            context = gram[:-1]
            context_count = self.unigram_counts[context[-1]] if context else sum(self.unigram_counts.values())
            self.ngram_probs[gram] = (count + 1) / (context_count + self.V)

    def infer_next_token(self, tokens):
        # tokens: list of token ids
        for n in range(self.n, 1, -1):
            if len(tokens) >= n - 1:
                context = tuple(tokens[-(n-1):])
                candidates = [(gram[-1], prob) for gram, prob in self.ngram_probs.items() if gram[:-1] == context]
                if candidates:
                    return max(candidates, key=lambda x: x[1])[0]
        return None

    def generate(self, prompt, max_length=50):
        tokens = self.tokenizer.encode(prompt)
        out = tokens[:]
        for _ in range(max_length):
            next_token = self.infer_next_token(out)
            if not next_token:
                break
            out.append(next_token)
        return self.tokenizer.decode(out)

    def perplexity(self, sentences):
        tokenized = [self.tokenizer.encode(s.lower()) for s in sentences]
        N = sum(len(sent) for sent in tokenized)
        log_prob = 0
        for sent in tokenized:
            n_grams = list(nltk.ngrams(sent, self.n))
            for gram in n_grams:
                prob = self.ngram_probs.get(gram, 1 / (self.V + 1))
                log_prob += -math.log(prob)
        return math.exp(log_prob / N) if N > 0 else float('inf')

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'n': self.n, 'ngram_probs': {str(k): v for k, v in self.ngram_probs.items()},
                       'vocab': list(self.vocab)}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        obj = cls(model_path=path,order=ORDER, tokenizer=BpeTokenizer())
        obj.ngram_probs = {eval(k): v for k, v in data['ngram_probs'].items()}
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
    df = df.sample(n=min(10000, len(df))).reset_index(drop=True)  # limit for speed

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
        model = NgramGenerator.load(model_path, tokenizer)
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
