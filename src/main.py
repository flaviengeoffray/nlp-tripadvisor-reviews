import os
import random
from typing import Optional

import click
import numpy as np
import torch

from pipelines.training import train
from pipelines.evaluation import evaluate
from pipelines.generation import generate
from pipelines.utils import load_config


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.

    :param int seed: Seed value to set.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


@click.group()
@click.option("--config", required=True, type=click.Path(exists=True, path_type=str))
@click.option("--seed", type=int, default=42, show_default=True)
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], seed: int) -> None:
    """
    Command line interface for the training script.

    :param click.Context ctx: Click context.
    :param Config config: Path to the configuration file.
    :param int seed: Seed value for reproducibility.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    set_seed(seed)


@cli.command()
@click.pass_context
def launchtrain(
    ctx: click.Context,
) -> None:
    """
    Load training data from files.

    :param click.Context ctx: Click context.
    """
    train(ctx.obj["config"])


@cli.command(name="eval")
@click.pass_context
def launch_evaluate(ctx: click.Context) -> None:
    """
    Evaluate the model on the given dataset.

    :param click.Context ctx: Click context.
    """
    evaluate(ctx.obj["config"])


@cli.command(name="infer")
@click.option("-p", "--prompt", "prompt", required=True, help="Prompt for generation.")
@click.option(
    "-r", "--rate", "rate", required=True, type=int, help="Rate for generation."
)
@click.pass_context
def launch_infer(ctx: click.Context, prompt: str, rate: int) -> None:
    """
    Load prompts from the command line or a file.

    :param click.Context ctx: Click context.
    :param str prompt: Prompt provided via command line.
    :param int rate: Rate provided via command line.
    """
    generate(ctx.obj["config"], prompt, rate)


if __name__ == "__main__":
    cli()
