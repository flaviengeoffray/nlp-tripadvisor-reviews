from collections import Counter
from nltk.tokenize import word_tokenize
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config import MAX_SEQUENCE_LENGTH, MIN_WORD_FREQ

class WordLevelReviewDataset(Dataset):
    def __init__(self, dataframe, vocab=None, max_seq_length=MAX_SEQUENCE_LENGTH):
        self.data = dataframe
        self.max_seq_length = max_seq_length
        
        # Special tokens
        self.PAD_TOKEN = "<PAD>" # Padding token for fixed-length sequences
        self.UNK_TOKEN = "<UNK>" # Unknown token for out-of-vocabulary words
        self.START_TOKEN = "<START>" # Start token for sequences
        self.END_TOKEN = "<END>" # End token for sequence
        
        if vocab is None:
            self.build_vocab()
        else:
            self.word2idx = vocab
            
        # Create reverse mapping for decoding
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def build_vocab(self):
        # Initialize with special tokens    
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.START_TOKEN: 2,
            self.END_TOKEN: 3
        }
        
        # Count word frequencies
        word_counts = Counter()
        for _, row in self.data.iterrows():
            # Clean and tokenize review
            words = self.tokenize(row['review'])
            word_counts.update(words)
        
        # Add words that appear frequently enough
        idx = 4  # Start after special tokens
        for word, count in word_counts.items():
            if count >= MIN_WORD_FREQ and word not in self.word2idx:
                self.word2idx[word] = idx
                idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)} words")

    def tokenize(self, text):
        """Clean and tokenize text into words"""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Tokenize into words
        words = word_tokenize(text)
        return words

    def encode(self, text):
        """Convert text to sequence of word indices"""
        words = self.tokenize(text)
        # Limit sequence length and add start/end tokens
        words = [self.START_TOKEN] + words[:self.max_seq_length-2] + [self.END_TOKEN]
        return [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]
    
    def decode(self, indices):
        """Convert sequence of indices back to text"""
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
        # Remove special tokens
        words = [word for word in words if word not in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]]
        return ' '.join(words)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        title_encoded = self.encode(row['title'])
        review_encoded = self.encode(row['review'])
        
        features = torch.tensor([
            row['overall'], row['cleanliness'], row['value'],
            row['location'], row['rooms'], row['sleep_quality']
        ], dtype=torch.float32) / 5.0  # Normalize to [0,1]
        
        return {
            'features': features,
            'title': torch.tensor(title_encoded, dtype=torch.long),
            'review': torch.tensor(review_encoded, dtype=torch.long)
        }

def collate_fn(batch):
    """
    Custom collate function for handling variable-length sequences in a batch.
    This function processes a batch of data to prepare it for input into a model. 
    It performs the following steps:
    - Stacks the 'features' tensors from each item in the batch.
    - Pads the 'title' and 'review' sequences to ensure uniform length across the batch.
    - Ensures the resulting tensors are stored in contiguous memory for efficient computation.
    - Splits the padded 'review' sequences into input (all tokens except the last) 
      and target (all tokens except the first) sequences for training.
    Args:
        batch (list of dict): A batch of data where each item is a dictionary containing:
            - 'features' (torch.Tensor): Feature tensor for the item.
            - 'title' (torch.Tensor): Title sequence tensor for the item.
            - 'review' (torch.Tensor): Review sequence tensor for the item.
    Returns:
        tuple: A tuple containing:
            - batch_features (torch.Tensor): Stacked feature tensors for the batch.
            - batch_titles (torch.Tensor): Padded title sequences for the batch.
            - reviews_input (torch.Tensor): Input sequences for the reviews (all but last token).
            - reviews_target (torch.Tensor): Target sequences for the reviews (all but first token).
    # Utility: This function is essential for preparing variable-length sequence data 
    # for training models, ensuring uniformity and efficiency in batch processing.
    """

    batch_features = torch.stack([item['features'] for item in batch])
    batch_titles = nn.utils.rnn.pad_sequence([item['title'] for item in batch], batch_first=True, padding_value=0)
    batch_reviews = nn.utils.rnn.pad_sequence([item['review'] for item in batch], batch_first=True, padding_value=0)
    
    # Ensure contiguous memory
    batch_features = batch_features.contiguous()
    batch_titles = batch_titles.contiguous()
    batch_reviews = batch_reviews.contiguous()
    
    # Create input (all but last token) and target (all but first token) sequences
    reviews_input = batch_reviews[:, :-1].contiguous()
    reviews_target = batch_reviews[:, 1:].contiguous()
    
    return batch_features, batch_titles, reviews_input, reviews_target
