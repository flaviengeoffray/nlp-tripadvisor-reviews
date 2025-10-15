import argparse
import warnings
import logging
from data.dataprep import prepare_data
from models.base import BaseModel
from .utils import load_tokenizer, load_vectorizer, load_model
from .config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train(config: Config) -> None:
    train_df, val_df, test_df = prepare_data(
        dataset_name=config.dataset_name,
        label_col=config.label_col,
        test_size=config.test_size,
        val_size=config.val_size,
        seed=config.seed,
        stratify=config.stratify,
        sample_size=config.sample_size,
        balance=config.balance,
        balance_percentage=config.balance_percentage,
        augmentation_methods=config.augmentation_methods,
        augmentation_workers=config.augmentation_workers,
        augmented_data=config.augmented_data,
    )

    X_train, y_train = (
        train_df[config.review_col].to_numpy(),
        train_df[config.label_col].to_numpy(),
    )

    X_val, y_val = (
        val_df[config.review_col].to_numpy(),
        val_df[config.label_col].to_numpy(),
    )

    X_test, y_test = (
        test_df[config.review_col].to_numpy(),
        test_df[config.label_col].to_numpy(),
    )

    if config.tokenizer:
        logger.info("Loading tokenizer...")
        tokenizer = load_tokenizer(config.tokenizer)
        if config.tokenizer.checkpoint:
            logger.info(
                f"Loading tokenizer checkpoint from {config.tokenizer.checkpoint}..."
            )
            tokenizer.load(str(config.tokenizer.checkpoint))
        else:
            logger.info("Fitting tokenizer on training data...")
            tokenizer.fit(X_train)
            logger.info(f"Saving tokenizer to {config.model_path / 'tokenizer.json'}...")
            
    if config.vectorizer:
        logger.info("Loading vectorizer...")
        vectorizer = load_vectorizer(config.vectorizer)

        if config.vectorizer.checkpoint:
            logger.info(
                f"Loading vectorizer checkpoint from {config.vectorizer.checkpoint}..."
            )
            vectorizer.load(config.vectorizer.checkpoint)
        else:
            logger.info("Fitting vectorizer on training data...")
            vectorizer.fit(X_train)
            logger.info(f"Saving vectorizer to {config.model_path / 'vectorizer.bz2'}...")

        logger.info("Transforming training and validation data with vectorizer...")
        X_train = vectorizer.transform(X_train)
        X_val = vectorizer.transform(X_val)
        X_test = vectorizer.transform(X_test)
            

    logger.info("Loading model...")
    model: BaseModel = load_model(config.model, config.model_path)
    
    model.tokenizer = tokenizer if config.tokenizer else None
    model.vectorizer = vectorizer if config.vectorizer else None

    if config.model.checkpoint:
        logger.info(f"Loading model checkpoint from {config.model.checkpoint}...")
        model.load(config.model.checkpoint)

    logger.info("Fitting model...")
    model.fit(X_train, y_train, X_val, y_val)

    logger.info("Evaluating model on test data...")
    metrics = model.evaluate(
        X_test,
        y_test,
        None,
    )
    logger.info("Metrics:", metrics)
