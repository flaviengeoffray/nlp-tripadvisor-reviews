import pandas as pd
import torch
from torch.utils.data import Dataset


class TripAdvisorDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        review_col: str = "review",
        label_col: str = "overall",
        tokenizer=None,
    ):
        self.texts = df[review_col].tolist()
        self.labels = df[label_col].tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        label = (float(label) - 1.0) / 4.0
        label = torch.tensor(label, dtype=torch.float)
        if self.tokenizer:
            enc = self.tokenizer(
                text, truncation=True, padding="max_length", return_tensors="pt"
            )
            enc = {k: v.squeeze(0) for k, v in enc.items()}
            return enc, torch.tensor(label, dtype=torch.long)
        return text, label
