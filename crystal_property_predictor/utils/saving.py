"""Functions that construct paths for saving experiment details."""
import datetime
from pathlib import Path

LOG_DIR: str = "logs"
CHECKPOINT_DIR: str = "checkpoints"
RUN_DIR: str = "tensorboard_summaries"


def ensure_exists(p: Path) -> Path:
    """Help to ensure a directory exists."""
    p: Path = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def arch_path(config: dict) -> Path:
    """Construct a path based on the experiment configuration's name.

    e.g. 'saved/EfficientNet/'
    """
    p: Path = Path(config["save_dir"]) / config["name"]
    return ensure_exists(p)


def arch_datetime_path(config: dict) -> Path:
    """Create a timestamped directory for experiment's single run."""
    start_time: str = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    p: Path = arch_path(config) / start_time
    return ensure_exists(p)


def log_path(config: dict) -> Path:
    """Create logs directory for a single experiment's configuration.

    Experiment's configuration is defined by its name,
    e.g. for a YAML file, the topmost line:
    name: EfficientNet
    """
    p: Path = arch_path(config) / LOG_DIR
    return ensure_exists(p)


def trainer_paths(config: dict) -> tuple[Path, Path]:
    """
    Return the paths to save checkpoints and tensorboard summaries.

    e.g.
    .. code::

        saved/EfficientNet/2023-10-02-123456/checkpoints/
        saved/EfficientNet/2023-10-02-123456/tensorboard_summaries/
    """
    arch_datetime: Path = arch_datetime_path(config)
    return (
        ensure_exists(arch_datetime / CHECKPOINT_DIR),
        ensure_exists(arch_datetime / RUN_DIR),
    )
