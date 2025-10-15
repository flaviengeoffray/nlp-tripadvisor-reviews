import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Dict, List

from my_tokenizers.base import BaseTokenizer


class TextDataset(Dataset):
    def __init__(
        self, texts: List[str], labels: List[int], tokenizer, max_len: int = 512
    ) -> None:
        self.texts: List[str] = texts
        self.labels: List[int] = labels
        self.tokenizer: BaseTokenizer = tokenizer
        self.max_len: int = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        text: str = self.texts[idx]
        tokens: List[int] = self.tokenizer.encode(text)
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]
        length: int = len(tokens)
        tokens = tokens + [self.tokenizer.token_to_id("[PAD]")] * (
            self.max_len - len(tokens)
        )

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(self.labels[idx] - 1, dtype=torch.long),
        }
