import argparse
import warnings
from data.dataprep import prepare_data
from models.base import BaseModel
from utils import load_config, load_tokenizer, load_vectorizer, load_model
import json

def main(config_path: str):
    config = load_config(config_path)

    train_df, val_df, test_df = prepare_data(
        dataset_name=config.dataset_name,
        label_col=config.label_col,
        drop_columns=["stay_year", "post_date", "freq", "lang"],
        test_size=config.test_size,
        val_size=config.val_size,
        seed=config.seed,
        stratify=config.stratify,
        sample_size=config.sample_size,
    )

    X_train, y_train = (
        train_df[config.review_col].to_numpy(),
        train_df[config.label_col].to_numpy(),
    )

    X_val, y_val = (
        val_df[config.review_col].to_numpy(),
        val_df[config.label_col].to_numpy(),
    )

    if config.tokenizer:
        tokenizer = load_tokenizer(config.tokenizer)
        if config.tokenizer.checkpoint:
            tokenizer.load(str(config.tokenizer.checkpoint))
        else:
            tokenizer.fit(X_train)
            tokenizer.save(str(config.model_path / "tokenizer.json"))

        config.model.params["tokenizer"] = tokenizer

    if config.vectorizer:
        vectorizer = load_vectorizer(config.vectorizer)

        if config.vectorizer.checkpoint:
            vectorizer.load(config.vectorizer.checkpoint)
        else:
            vectorizer.fit(X_train)
            vectorizer.save(config.model_path / "vectorizer.bz2")

        X_train = vectorizer.transform(X_train)
        X_val = vectorizer.transform(X_val)

    model: BaseModel = load_model(config.model, config.model_path)

    if config.model.checkpoint:
        model.load(config.model.checkpoint)

    model.fit(X_train, y_train, X_val, y_val)


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
