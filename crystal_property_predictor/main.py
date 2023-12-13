import os
import random
from logging import Logger
from types import ModuleType
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler
from torch.utils.data import DataLoader

import crystal_property_predictor.data_loader.augmentations as module_aug
import crystal_property_predictor.data_loader.cross_validators as module_cv
import crystal_property_predictor.data_loader.data_loaders as module_data
import crystal_property_predictor.model.losses as module_loss
import crystal_property_predictor.model.metrics as module_metric
import crystal_property_predictor.model.models as module_arch
from crystal_property_predictor.base import (
    AugmentationFactoryBase,
    CrossValidatorFactoryBase,
    DataLoaderBase,
)
from crystal_property_predictor.trainer import Trainer
from crystal_property_predictor.utils import setup_logger

log: Logger = setup_logger(__name__)


def train(cfg: dict, resume: str | None) -> None:
    do_cross_validation: bool = cfg["training"]["do_cross_validation"]

    transforms: AugmentationFactoryBase | None = get_instance(
        module_aug, "augmentation", cfg
    )
    cross_validator: CrossValidatorFactoryBase | None = get_instance(
        module_cv, "cross_validation", cfg
    )
    data_loader: DataLoaderBase = get_instance(
        module_data, "data_loader", cfg, transforms, cross_validator
    )
    valid_data_loader: DataLoader[Any] | None = data_loader.split_validation()

    if do_cross_validation:
        train_cross_validation(cfg, resume, data_loader)
    else:
        train_normal(cfg, resume, data_loader, valid_data_loader)


def train_cross_validation(
    cfg: dict,
    resume: str | None,
    data_loader: DataLoaderBase,
) -> None:
    cv_folds_generator = data_loader.generate_cross_validation_folds()
    fold_no = 0

    while True:
        try:
            train_fold_loader, valid_fold_loader = next(cv_folds_generator)
        except StopIteration:
            break

        fold_no += 1

        log.info("----------------------------------------")
        log.info(f"Fold {fold_no}")
        log.info("----------------------------------------")

        train_normal(cfg, resume, train_fold_loader, valid_fold_loader)


def train_normal(
    cfg: dict,
    resume: str | None,
    data_loader: DataLoaderBase,
    valid_data_loader: DataLoader[Any] | None,
) -> None:
    log.debug(f"Training: {cfg}")
    seed_everything(cfg["seed"])

    model: nn.Module = get_instance(module_arch, "arch", cfg)
    device: torch.device
    model, device = setup_device(model, cfg["target_devices"])
    torch.backends.cudnn.benchmark = True  # disable if not consistent input sizes

    param_groups: list[dict] = setup_param_groups(model, cfg["optimizer"])
    optimizer: torch.optim.Optimizer = get_instance(
        module_optimizer, "optimizer", cfg, param_groups
    )
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = get_instance(
        module_scheduler, "lr_scheduler", cfg, optimizer
    )
    start_epoch: int
    model, optimizer, start_epoch = resume_checkpoint(resume, model, optimizer, cfg)

    log.info("Getting loss and metric function handles")
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(
        module_loss, cfg["loss"]
    )
    metrics: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float]] = [
        getattr(module_metric, met) for met in cfg["metrics"]
    ]

    log.info("Initialising trainer")
    trainer: Trainer = Trainer(
        model,
        loss,
        metrics,
        optimizer,
        start_epoch=start_epoch,
        config=cfg,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()
    log.info("Finished!")


def setup_device(
    model: nn.Module, target_devices: list[int]
) -> tuple[nn.Module, torch.device]:
    """Set up GPU device if available, move model into configured device."""
    available_devices: list[int] = list(range(torch.cuda.device_count()))
    device: torch.device

    if not available_devices:
        log.warning(
            "There's no GPU available on this machine. Training will be performed on "
            "CPU."
        )
        device = torch.device("cpu")
        model = model.to(device)
        return model, device

    if not target_devices:
        log.info("No GPU selected. Training will be performed on CPU.")
        device = torch.device("cpu")
        model = model.to(device)
        return model, device

    max_target_gpu: int = max(target_devices)
    max_available_gpu: int = max(available_devices)

    if max_target_gpu > max_available_gpu:
        msg: str = (
            f"Configuration requests GPU #{max_target_gpu} but only {max_available_gpu}"
            f" available. Check the configuration and try again."
        )
        log.critical(msg)
        raise Exception(msg)

    log.info(
        f"Using devices {target_devices} of available devices " f"{available_devices}"
    )
    device = torch.device(f"cuda:{target_devices[0]}")
    if len(target_devices) > 1:
        model = nn.DataParallel(model, device_ids=target_devices).to(device)
    else:
        model = model.to(device)
    return model, device


def setup_param_groups(model: nn.Module, config: dict) -> list[dict]:
    return [{"params": model.parameters(), **config}]


def resume_checkpoint(
    resume_path: str | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> tuple[nn.Module, torch.optim.Optimizer, int]:
    """Resume from saved checkpoint."""
    if resume_path is None:
        start_epoch_: int = 1
        return model, optimizer, start_epoch_

    log.info(f"Loading checkpoint: {resume_path}")
    checkpoint: Any = torch.load(resume_path)
    model.load_state_dict(checkpoint["state_dict"])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if checkpoint["config"]["optimizer"]["type"] != config["optimizer"]["type"]:
        log.warning(
            "Warning: Optimizer type given in config file is different from that of "
            "checkpoint. Optimizer parameters not being resumed."
        )
    else:
        optimizer.load_state_dict(checkpoint["optimizer"])

    log.info(f'Checkpoint "{resume_path}" loaded')
    return model, optimizer, checkpoint["epoch"]


def get_instance(module: ModuleType, name: str, config: dict, *args: Any) -> Any:
    """Help to construct an instance of a class.

    Parameters
    ----------
    module : ModuleType
        Module containing the class to construct.
    name : str
        Name of class, as would be returned by ``.__class__.__name__``.
    config : dict
        Dictionary containing an 'args' item, which will be used as
        ``kwargs`` to construct the
        class instance.
    args : Any
        Positional arguments to be given before ``kwargs`` in ``config``.
    """
    ctor_name = config[name]["type"]
    log.info(f"Building: {module.__name__}.{ctor_name}")
    return getattr(module, ctor_name)(*args, **config[name]["args"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
