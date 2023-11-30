from logging import Logger
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn as nn
from overrides import override
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from crystal_property_predictor.base import DataLoaderBase, TrainerBase
from crystal_property_predictor.utils import AverageMeter, setup_logger

log: Logger = setup_logger(__name__)


class Trainer(TrainerBase):
    """Responsible for training loop and validation."""

    @override
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float],
        optimizer: torch.optim.Optimizer,
        start_epoch: int,
        config: dict,
        device: torch.device,
        data_loader: DataLoaderBase,
        valid_data_loader: DataLoader[Any] | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        super().__init__(model, loss, metrics, optimizer, start_epoch, config, device)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation: bool = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step: int = int(np.sqrt(data_loader.batch_size)) * 8

    @override
    def _train_epoch(self, epoch: int) -> dict:
        """Training logic for an epoch.

        Returns
        -------
        dict
            Dictionary containing results for the epoch.
        """
        self.model.train()

        loss_mtr: AverageMeter = AverageMeter("loss")
        metric_mtrs: list[AverageMeter] = [
            AverageMeter(m.__name__) for m in self.metrics
        ]

        batch_idx: int
        data: torch.Tensor
        target: torch.Tensor
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output: torch.Tensor = self.model(data)
            loss: torch.Tensor = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            loss_mtr.update(loss.item(), data.size(0))

            if batch_idx % self.log_step == 0:
                self.writer.set_step(epoch * len(self.data_loader) + batch_idx)
                self.writer.add_scalar("batch/loss", loss.item())
                mtr: AverageMeter
                value: torch.Tensor
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))
                    self.writer.add_scalar(f"batch/{mtr.name}", value)
                self._log_batch(
                    epoch,
                    batch_idx,
                    self.data_loader.batch_size,
                    len(self.data_loader),
                    loss.item(),
                )

            if batch_idx == 0:
                self.writer.add_image(
                    "data", make_grid(data.cpu(), nrow=8, normalize=True)
                )

        del data
        del target
        del output
        torch.cuda.empty_cache()

        self.writer.add_scalar("epoch/loss", loss_mtr.avg)
        mtr_: AverageMeter
        for mtr_ in metric_mtrs:
            self.writer.add_scalar(f"epoch/{mtr_.name}", mtr_.avg)

        results: dict = {
            "loss": loss_mtr.avg,
            "metrics": [mtr.avg for mtr in metric_mtrs],
        }

        if self.do_validation:
            val_results: dict = self._valid_epoch(epoch)
            results = {**results, **val_results}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return results

    def _log_batch(
        self, epoch: int, batch_idx: int, batch_size: int, len_data: int, loss: float
    ) -> None:
        n_samples: int = batch_size * len_data
        n_complete: int = batch_idx * batch_size
        percent: float = 100.0 * batch_idx / len_data
        msg: str = (
            f"Train Epoch: {epoch} [{n_complete}/{n_samples} ("
            f"{percent:.0f}%)] Loss: {loss:.6f}"
        )
        log.debug(msg)

    def _eval_metrics(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> Iterator[torch.Tensor | float]:
        with torch.no_grad():
            metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float]
            for metric in self.metrics:
                value: torch.Tensor | float
                value = metric(output, target)
                yield value

    def _valid_epoch(self, epoch: int) -> dict:
        """Validate after training an epoch.

        Returns
        -------
        dict
            Contains keys 'val_loss' and 'val_metrics'.
        """
        self.model.eval()
        loss_mtr: AverageMeter = AverageMeter("loss")
        metric_mtrs: list[AverageMeter] = [
            AverageMeter(m.__name__) for m in self.metrics
        ]
        with torch.no_grad():
            batch_idx: int
            data: torch.Tensor
            target: torch.Tensor
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss(output, target)
                loss_mtr.update(loss.item(), data.size(0))
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))
                if batch_idx == 0:
                    self.writer.add_image(
                        "input", make_grid(data.cpu(), nrow=8, normalize=True)
                    )

        del data
        del target
        del output
        torch.cuda.empty_cache()

        self.writer.set_step(epoch, "valid")
        self.writer.add_scalar("loss", loss_mtr.avg)
        mtr_: AverageMeter
        for mtr_ in metric_mtrs:
            self.writer.add_scalar(mtr_.name, mtr_.avg)

        return {
            "val_loss": loss_mtr.avg,
            "val_metrics": [mtr.avg for mtr in metric_mtrs],
        }
