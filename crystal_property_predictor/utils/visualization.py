from pathlib import Path
from typing import Callable, Any

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError(
        "Import `from torch.utils.tensorboard import SummaryWriter` failed."
        "Ensure PyTorch version >= 1.1 and Tensorboard > 1.14 are installed."
    )


class TensorboardWriter:
    def __init__(self, writer_dir: Path, enabled: bool):
        self.writer: SummaryWriter | None = None
        if enabled:
            self.writer = SummaryWriter(str(writer_dir))

        self.step: int = 0
        self.mode: str = ""

        self.tb_writer_ftns: list[str] = [
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        ]
        self.tag_mode_exceptions: list[str] = ["add_histogram", "add_embedding"]

    def set_step(self, step, mode="train") -> None:
        self.mode = mode
        self.step = step

    def __getattr__(self, name: str) -> Callable[[Any], None]:
        """.

        If visualization is configured to use:
            return add_data() methods of tensorboard with additional
            information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data: Callable[[Any], None] | None = getattr(self.writer, name, None)

            def wrapper(tag: str, data, *args, **kwargs) -> None:
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = f"{self.mode}/{tag}"
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class,
            # set_step() for instance.
            try:
                attr: Callable[[Any], None] = getattr(object, name)
            except AttributeError:
                raise AttributeError(
                    f"type object `TensorboardWriter` has no attribute {name}"
                )
            return attr
