from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Optional

from .datasets.TripAdvisorDataset import TripAdvisorDataset


def prepare_data(
    dataset_name: str = "jniimi/tripadvisor-review-rating",
    label_col: str = "overall",
    drop_columns: list[str] = ["stay_year", "post_date", "freq", "lang"],
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
    sample_size: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    raw = load_dataset(dataset_name)
    df = pd.DataFrame(raw["train"])

    df = df.drop(columns=drop_columns, errors="ignore")
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if sample_size is not None:
        print("df sample ", sample_size)
        df = df.sample(n=sample_size, random_state=seed)

    if stratify:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df[label_col],
        )

        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=seed,
            stratify=train_df[label_col],
        )
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        train_df, val_df = train_test_split(
            train_df, test_size=val_size, random_state=seed
        )

    return train_df, val_df, test_df


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    tokenizer=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TripAdvisorDataset(train_df, tokenizer)
    val_ds = TripAdvisorDataset(val_df, tokenizer)
    test_ds = TripAdvisorDataset(test_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
