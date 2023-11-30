from pathlib import Path
import datetime

LOG_DIR: str = "logs"
CHECKPOINT_DIR: str = "checkpoints"
RUN_DIR: str = "runs"


def ensure_exists(p: Path) -> Path:
    """Help to ensure a directory exists."""
    p: Path = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def arch_path(config: dict) -> Path:
    """Construct a path based on the name of a configuration file.

    e.g. 'saved/EfficientNet'
    """
    p: Path = Path(config["save_dir"]) / config["name"]
    return ensure_exists(p)


def arch_datetime_path(config: dict) -> Path:
    start_time: str = datetime.datetime.now().strftime("%m%d-%H%M%S")
    p: Path = arch_path(config) / start_time
    return ensure_exists(p)


def log_path(config: dict) -> Path:
    p: Path = arch_path(config) / LOG_DIR
    return ensure_exists(p)


def trainer_paths(config: dict) -> tuple[Path, Path]:
    """
    Return the paths to save checkpoints and tensorboard runs.

    e.g.
    .. code::

        saved/EfficientNet/1002-123456/checkpoints
        saved/EfficientNet/1002-123456/runs
    """
    arch_datetime: Path = arch_datetime_path(config)
    return (
        ensure_exists(arch_datetime / CHECKPOINT_DIR),
        ensure_exists(arch_datetime / RUN_DIR),
    )
