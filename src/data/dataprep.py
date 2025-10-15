import logging
from typing import Optional

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .augmentation import DataAugmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def prepare_data(
    dataset_name: str = "jniimi/tripadvisor-review-rating",
    label_col: str = "overall",
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
    """
    Load and prepare dataset for training, validation, and testing.

    :param str dataset_name: Name of the dataset to load.
    :param str label_col: Name of the label column.
    :param float test_size: Proportion of the dataset to include in the test split.
    :param float val_size: Proportion of the training set to include in the validation split
    :param int seed: Random seed for reproducibility.
    :param bool stratify: Whether to stratify splits based on the label column.
    :param Optional[int] sample_size: If provided, sample this many instances from the dataset
    :param bool balance: Whether to balance the dataset using data augmentation.
    :param float balance_percentage: Target percentage of the majority class after balancing.
    :param List[str] augmentation_methods: List of augmentation methods to use for balancing.
    :param Optional[int] augmentation_workers: Number of parallel workers for augmentation.
    :param Optional[str] augmented_data: Path to CSV file with pre-augmented data
    :return tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test dataframes.
    """
    logger.info("Loading dataset...")
    raw = load_dataset(dataset_name)
    df = pd.DataFrame(raw["train"])

    if augmented_data is not None:
        df_aug = pd.read_csv(augmented_data)
        df = pd.concat([df, df_aug], ignore_index=True)
    logger.info("Dataset loaded.")

    if balance:
        logger.info("Balancing dataset...")

        logger.info("Class distribution before balancing:")
        logger.info(df[label_col].value_counts())

        augmenter = DataAugmentation(
            random_state=seed, num_workers=augmentation_workers
        )
        logger.info("Augmentation methods: %s", augmentation_methods)

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

        logger.info("Class distribution after balancing:")
        logger.info(df[label_col].value_counts())
    else:
        logger.info("Class distribution:")
        logger.info(df[label_col].value_counts())

    if sample_size is not None:
        logger.info("Sample size: %d", sample_size)
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df[label_col], random_state=seed
        )
        logger.info("Class distribution of subset:")
        logger.info(df[label_col].value_counts())

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
