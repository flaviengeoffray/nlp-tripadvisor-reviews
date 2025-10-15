from models.base import BaseModel
from models.generative.base import BaseGenerativeModel
from .utils import load_tokenizer, load_model
from .config import Config


def generate(config: Config, prompt: str, rate: int) -> None:
    """
    Generate text based on the provided prompt and rate.

    :param Config config: Configuration object.
    :param str prompt: Prompt for text generation.
    :param int rate: Rate for text generation.
    """
    if config.tokenizer:
        print("Loading tokenizer...")
        tokenizer = load_tokenizer(config.tokenizer)
        if config.tokenizer.checkpoint:
            print(f"Loading tokenizer checkpoint from {config.tokenizer.checkpoint}...")
            tokenizer.load(str(config.tokenizer.checkpoint))

        config.model.params["tokenizer"] = tokenizer

    print("Loading model...")
    model: BaseModel = load_model(config.model, config.model_path)
    if not isinstance(model, BaseGenerativeModel):
        raise ValueError("Model must be an instance of BaseGenerativeModel.")

    if not config.model.checkpoint:
        raise Exception("Model checkpoint is needed for evaluation.")

    print(f"Loading model checkpoint from {config.model.checkpoint}...")
    model.load(config.model.checkpoint)

    print("Generating model on test data...")

    generation = model.generate(
        f"{float(rate)}: {prompt}"  # float conversion to have 5.0 instead of 5, data was trained with floats
    )
    print("Generation:", generation)
