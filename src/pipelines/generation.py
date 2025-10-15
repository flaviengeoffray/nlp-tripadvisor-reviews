from models.base import BaseModel
from typing import List
from models.generative.base import BaseGenerativeModel
from .utils import load_tokenizer, load_model
from .config import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def generate(config: Config, prompt: str, rate: int) -> None:
    """
    Generate text based on the provided prompt and rate.

    :param Config config: Configuration object.
    :param str prompt: Prompt for text generation.
    :param int rate: Rate for text generation.
    """
    if config.tokenizer:
        logger.info("Loading tokenizer...")
        tokenizer = load_tokenizer(config.tokenizer)
        if config.tokenizer.checkpoint:
            logger.info(f"Loading tokenizer checkpoint from {config.tokenizer.checkpoint}...")
            tokenizer.load(str(config.tokenizer.checkpoint))

        config.model.params["tokenizer"] = tokenizer

    logger.info("Loading model...")
    model: BaseModel = load_model(config.model, config.model_path)
    if not isinstance(model, BaseGenerativeModel):
        raise ValueError("Model must be an instance of BaseGenerativeModel.")

    if not config.model.checkpoint:
        raise Exception("Model checkpoint is needed for evaluation.")

    logger.info(f"Loading model checkpoint from {config.model.checkpoint}...")
    model.load(config.model.checkpoint)

    logger.info("Generating content...")
    
    generation: List[str] = model.generate(
        f"{float(rate)}: {prompt}"  # float conversion to have 5.0 instead of 5, data was trained with floats
    )

    logger.info("Generated content: %s", " ".join(generation))
