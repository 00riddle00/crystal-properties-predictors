"""Functions that set up logging configuration and produce loggers."""
import logging
import logging.config
from pathlib import Path
from typing import TextIO

import yaml

from .saving import log_path

LOG_LEVEL: int = logging.INFO


def setup_logging(run_config: dict, log_config_: str = "logging.yml") -> None:
    """Set up ``logging.config``, i.e. modify default logging configuration.

    Parameters
    ----------
    run_config : Configuration for experiment's single run

    log_config_ : Path to configuration file for logging
    """
    log_config: Path = Path(log_config_)
    if not log_config.exists():
        logging.basicConfig(level=LOG_LEVEL)
        logger: logging.Logger = logging.getLogger("setup")
        logger.warning(f"'{log_config}' not found. Using basicConfig.")
        return

    f: TextIO
    with open(log_config, "rt") as f:
        config: dict = yaml.safe_load(f.read())

    # Create logging paths based on run config.
    run_path: Path = log_path(run_config)

    handler_name: str
    handler: dict
    for handler_name, handler in config["handlers"].items():
        if "filename" in handler:
            handler["filename"] = str(run_path / handler["filename"])

    logging.config.dictConfig(config)


def setup_logger(module_name: str) -> logging.Logger:
    """Create a logger, will be consumed by Python modules."""
    logger: logging.Logger = logging.getLogger(
        f"crystal_property_predictor.{module_name}"
    )
    logger.setLevel(LOG_LEVEL)
    return logger
