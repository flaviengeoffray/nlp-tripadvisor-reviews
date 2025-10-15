import argparse
import warnings
from data.dataprep import prepare_data
from models.base import BaseModel
from .utils import load_tokenizer, load_vectorizer, load_model
from .config import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate(config: Config) -> None:
    """
    Load and prepare the test dataset for evaluation.

    :param Config config: Configuration object.
    """
    _, _, test_df = prepare_data(
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

        config.model.params["tokenizer"] = tokenizer

    if config.vectorizer:
        logger.info("Loading vectorizer...")
        vectorizer = load_vectorizer(config.vectorizer)

        if config.vectorizer.checkpoint:
            logger.info(
                f"Loading vectorizer checkpoint from {config.vectorizer.checkpoint}..."
            )
            vectorizer.load(config.vectorizer.checkpoint)

        logger.info("Transforming data with vectorizer...")
        X_test = vectorizer.transform(X_test)

    logger.info("Loading model...")
    model: BaseModel = load_model(config.model, config.model_path)

    if not config.model.checkpoint:
        raise Exception("Model checkpoint is needed for evaluation.")

    logger.info(f"Loading model checkpoint from {config.model.checkpoint}...")
    model.load(config.model.checkpoint)

    logger.info("Evaluating model on test data...")
    metrics = model.evaluate(
        X_test,
        y_test,
        None,
    )
    logger.info("Metrics:", metrics)
