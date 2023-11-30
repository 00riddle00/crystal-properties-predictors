from typing import TextIO

import click
import yaml

from crystal_property_predictor import main
from crystal_property_predictor.utils import setup_logging


@click.group()
def cli():
    """CLI for crystal_property_predictor."""
    pass


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default=["experiments/config.yml"],  # default must be provided
    multiple=True,
    help=(
        "Path to training configuration file. If multiple are provided, runs will be "
        "executed in order"
    ),
)
@click.option("-r", "--resume", default=None, type=str, help="path to checkpoint")
def train(config_filename: tuple[str], resume: str | None):
    """Entry point to start training run(s)."""
    configs: list[dict] = [load_config(f) for f in config_filename]
    config: dict
    for config in configs:
        setup_logging(config)
        main.train(config, resume)


def load_config(filename: str) -> dict:
    """Load a configuration file as YAML."""
    fh: TextIO
    with open(filename) as fh:
        config: dict = yaml.safe_load(fh)
        if config is None:
            raise ValueError(f"Config file {filename} is empty")
    return config
