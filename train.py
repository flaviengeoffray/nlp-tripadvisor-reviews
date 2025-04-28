import argparse
from data.dataprep import prepare_data
from models.base import BaseModel
from utils import load_config, load_tokenizer, load_vectorizer, load_model


def main(config_path: str):
    config = load_config(config_path)

    train_df, val_df, test_df = prepare_data(dataset_name=config.dataset_name)
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
            tokenizer.load(config.tokenizer.checkpoint)
        else:
            tokenizer.fit(X_train)
        # Use tokenizer.encode then tokenizer.save to save it

    if config.vectorizer:
        vectorizer = load_vectorizer(config.vectorizer)

        if config.vectorizer.checkpoint:
            vectorizer.load(config.vectorizer.checkpoint)
        else:
            vectorizer.fit(X_train)

        X_train = vectorizer.transform(X_train)
        X_val = vectorizer.transform(X_val)

        vectorizer.save(config.model_path / "vectorizer.bz2")

    model: BaseModel = load_model(config.model, config.model_path)

    model.fit(X_train, y_train, X_val, y_val)

    model.save(config.model_path / "model.bz2")


if __name__ == "__main__":
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
