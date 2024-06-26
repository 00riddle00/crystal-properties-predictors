import math
from logging import Logger
from pathlib import Path
from typing import Callable, TextIO

import torch
import torch.nn as nn
import yaml

from crystal_property_predictor.utils import (
    TensorboardWriter,
    setup_logger,
    trainer_paths,
)

log: Logger = setup_logger(__name__)


class TrainerBase:
    """Base class for all trainers."""

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float],
        optimizer: torch.optim.Optimizer,
        start_epoch: int,
        config: dict,
        device: torch.device,
    ):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.config = config
        self.device = device

        # The following attributes will be set in _setup_monitoring() method
        self.epochs: int | None = None
        self.save_period: int | None
        self.monitor: str | None = None
        self.mnt_mode: str | None = None
        self.mnt_metric: str | None = None
        self.mnt_best: float | None = None
        self.early_stop: int | None = None
        self._setup_monitoring(config["training"])

        self.checkpoint_dir: Path
        writer_dir: Path
        self.checkpoint_dir, writer_dir = trainer_paths(config)
        self.writer: TensorboardWriter = TensorboardWriter(
            writer_dir, config["training"]["tensorboard"]
        )

        # Save configuration file into checkpoint directory:
        config_save_path: Path = Path(self.checkpoint_dir) / "config.yml"
        handle: TextIO
        with open(config_save_path, "w") as handle:
            yaml.dump(config, handle, default_flow_style=False)

    def train(self):
        """Full training logic."""
        log.info("Starting training...")
        # Counting epochs starts from 1
        epoch: int
        for epoch in range(self.start_epoch, self.epochs + 1):
            result: dict = self._train_epoch(epoch)

            # save logged information into log dict
            results: dict = {"epoch": epoch}
            key: str
            value: list[torch.Tensor] | float
            for key, value in result.items():
                if key == "metrics":
                    results.update(
                        {mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)}
                    )
                elif key == "val_metrics":
                    results.update(
                        {
                            "val_" + mtr.__name__: value[i]
                            for i, mtr in enumerate(self.metrics)
                        }
                    )
                else:
                    results[key] = value

            # print logged information to the screen
            key_: str
            value_: torch.Tensor | int | float
            for key_, value_ in results.items():
                log.info(f"{str(key_):15s}: {value_}")

            # evaluate model performance according to configured metric save
            # the best checkpoint as model_best
            best: bool = False
            if self.mnt_mode != "off":
                not_improved_count: int = 0
                improved: bool
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == "min"
                        and results[self.mnt_metric] < self.mnt_best
                    ) or (
                        self.mnt_mode == "max"
                        and results[self.mnt_metric] > self.mnt_best
                    )
                except KeyError:
                    log.warning(
                        f"Warning: Metric '{self.mnt_metric}' is not found. Model "
                        f"performance monitoring is disabled."
                    )
                    self.mnt_mode = "off"
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = results[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    log.info(
                        f"Validation performance didn't improve for {self.early_stop} "
                        f"epochs. Training stops."
                    )
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _train_epoch(self, epoch: int) -> dict:
        """Training logic for an epoch."""
        raise NotImplementedError

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        """Save checkpoints.

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to
        'model_best.pth'
        """
        arch: str = type(self.model).__name__
        state: dict = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename: Path = self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth"
        torch.save(state, filename)
        log.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path: Path = self.checkpoint_dir / "model_best.pth"
            torch.save(state, best_path)
            log.info(f"Saving current best: {best_path}")

    def _setup_monitoring(self, config: dict) -> None:
        """Configure to monitor model performance and save best."""
        self.epochs = config["epochs"]
        self.save_period = config["save_period"]
        self.monitor = config.get("monitor", "off")
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0.0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = math.inf if self.mnt_mode == "min" else -math.inf
            self.early_stop = config.get("early_stop", math.inf)
