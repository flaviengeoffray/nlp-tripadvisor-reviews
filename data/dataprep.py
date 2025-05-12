from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional

from .augmentation.DataAugmentation import DataAugmentation


def prepare_data(
    dataset_name: str = "jniimi/tripadvisor-review-rating",
    label_col: str = "overall",
    drop_columns: list[str] = ["stay_year", "post_date", "freq", "lang"],
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
    sample_size: Optional[int] = None,
    balance: bool = False,
    balance_percentage: float = 0.8,
    augmentation_methods: list[str] = ["synonym", "random"],
    augmentation_workers: Optional[int] = None,
    augmented_data: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    print("Loading dataset...")
    raw = load_dataset(dataset_name)
    df = pd.DataFrame(raw["train"])

    if augmented_data is not None:
        df_aug = pd.read_csv(augmented_data)
        df = pd.concat([df, df_aug], ignore_index=True)
    print("Dataset loaded.")

    if balance:
        print("Balancing dataset...")

        print("Class distribution before balancing:")
        print(df[label_col].value_counts())

        augmenter = DataAugmentation(
            random_state=seed, num_workers=augmentation_workers
        )
        print("Augmentation methods:", augmentation_methods)

        class_counts = df[label_col].value_counts().to_dict()
        max_count = max(class_counts.values())
        target_counts = {
            label: int(max_count * balance_percentage)
            for label in class_counts
            if label != 5
        }
        target_counts[5] = class_counts[5]
        df = augmenter.balance_dataset(
            df,
            text_col="review",
            label_col=label_col,
            target_counts=target_counts,
            methods=augmentation_methods,
        )

        print("Class distribution after balancing:")
        print(df[label_col].value_counts())
    else:
        print("Class distribution:")
        print(df[label_col].value_counts())

    if sample_size is not None:
        print("Sample size:", sample_size)
        # df = df.sample(n=sample_size, random_state=seed)
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df[label_col], random_state=seed
        )
        print("Class distribution of subset:")
        print(df[label_col].value_counts())

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

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
