import argparse
import warnings
from data.dataprep import prepare_data
from models.base import BaseModel
from utils import load_config, load_tokenizer, load_vectorizer, load_model


def main(config_path: str):
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)

    _, _, test_df = prepare_data(
        dataset_name=config.dataset_name,
        label_col=config.label_col,
        drop_columns=["stay_year", "post_date", "freq", "lang"],
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
        print("Loading tokenizer...")
        tokenizer = load_tokenizer(config.tokenizer)
        if config.tokenizer.checkpoint:
            print(f"Loading tokenizer checkpoint from {config.tokenizer.checkpoint}...")
            tokenizer.load(str(config.tokenizer.checkpoint))

        config.model.params["tokenizer"] = tokenizer

    if config.vectorizer:
        print("Loading vectorizer...")
        vectorizer = load_vectorizer(config.vectorizer)

        if config.vectorizer.checkpoint:
            print(
                f"Loading vectorizer checkpoint from {config.vectorizer.checkpoint}..."
            )
            vectorizer.load(config.vectorizer.checkpoint)

        print("Transforming data with vectorizer...")
        X_test = vectorizer.transform(X_test)

    print("Loading model...")
    model: BaseModel = load_model(config.model, config.model_path)

    if not config.model.checkpoint:
        raise Exception("Model checkpoint is needed for evaluation.")

    print(f"Loading model checkpoint from {config.model.checkpoint}...")
    model.load(config.model.checkpoint)

    print("Evaluating model on test data...")
    metrics = model.evaluate(
        X_test,
        y_test,
        None,
    )
    print("Metrics:", metrics)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Config path",
    )
    args = parser.parse_args()
    main(args.config)
