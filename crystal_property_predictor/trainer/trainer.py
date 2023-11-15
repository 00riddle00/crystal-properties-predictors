import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.utils import make_grid

from crystal_property_predictor.base import AverageMeter, TrainerBase
from crystal_property_predictor.utils import setup_logger

log = setup_logger(__name__)


class Trainer(TrainerBase):
    """Responsible for training loop and validation."""

    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        start_epoch,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
    ):
        super().__init__(model, loss, metrics, optimizer, start_epoch, config, device)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size)) * 8

        self.dataset = self.data_loader.train_dataset

    def train(self):
        """Full training logic."""
        log.info("Starting training...")

        k = 10
        splits = KFold(n_splits=k, shuffle=True, random_state=42)
        foldperf = {}

        for fold, (train_idx, val_idx) in enumerate(
            splits.split(np.arange(len(self.dataset)))
        ):
            log.info("----------------------------------------")
            log.info("Fold {}".format(fold + 1))
            log.info("----------------------------------------")
            not_improved_count = 0

            # init_kwargs = {"batch_size": 64, "num_workers": 2}

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(
                self.dataset, sampler=train_sampler, batch_size=64, num_workers=2
            )
            test_loader = DataLoader(
                self.dataset, sampler=test_sampler, batch_size=64, num_workers=2
            )

            self.data_loader = train_loader
            self.valid_data_loader = test_loader

            for epoch in range(self.start_epoch, self.epochs):
                result = self._train_epoch(epoch)

                # save logged information into log dict
                results = {"epoch": epoch}
                for key, value in result.items():
                    if key == "metrics":
                        results.update(
                            {
                                mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)
                            }
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
                for key, value in results.items():
                    log.info(f"{str(key):15s}: {value}")

                # evaluate model performance according to configured metric save
                # the best checkpoint as model_best
                best = False
                if self.mnt_mode != "off":
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
        """Training logic for an epoch.

        Returns
        -------
        dict
            Dictionary containing results for the epoch.
        """
        self.model.train()

        loss_mtr = AverageMeter("loss")
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            loss_mtr.update(loss.item(), data.size(0))

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch) * len(self.data_loader) + batch_idx)
                self.writer.add_scalar("batch/loss", loss.item())
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
        for mtr in metric_mtrs:
            self.writer.add_scalar(f"epoch/{mtr.name}", mtr.avg)

        results = {
            "loss": loss_mtr.avg,
            "metrics": [mtr.avg for mtr in metric_mtrs],
        }

        if self.do_validation:
            val_results = self._valid_epoch(epoch)
            results = {**results, **val_results}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return results

    def _log_batch(self, epoch, batch_idx, batch_size, len_data, loss):
        n_samples = batch_size * len_data
        n_complete = batch_idx * batch_size
        percent = 100.0 * batch_idx / len_data
        msg = (
            f"Train Epoch: {epoch} [{n_complete}/{n_samples} ("
            f"{percent:.0f}%)] Loss: {loss:.6f}"
        )
        log.debug(msg)

    def _eval_metrics(self, output, target):
        with torch.no_grad():
            for metric in self.metrics:
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
        loss_mtr = AverageMeter("loss")
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]
        with torch.no_grad():
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
        for mtr in metric_mtrs:
            self.writer.add_scalar(mtr.name, mtr.avg)

        return {
            "val_loss": loss_mtr.avg,
            "val_metrics": [mtr.avg for mtr in metric_mtrs],
        }
