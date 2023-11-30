import logging
import logging.config
from pathlib import Path
from typing import TextIO

import yaml

from .saving import log_path

LOG_LEVEL: int = logging.INFO


def setup_logging(run_config: dict, log_config_: str = "logging.yml") -> None:
    """Set up ``logging.config``.

    Parameters
    ----------
    run_config : Configuration for run

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

    # modify logging paths based on run config
    run_path: Path = log_path(run_config)

    tw_handler_name: str
    handler: dict
    for tw_handler_name, handler in config["handlers"].items():
        if "filename" in handler:
            handler["filename"] = str(run_path / handler["filename"])

    logging.config.dictConfig(config)


def setup_logger(name: str) -> logging.Logger:
    log: logging.Logger = logging.getLogger(f"crystal_property_predictor.{name}")
    log.setLevel(LOG_LEVEL)
    return log
