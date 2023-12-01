from typing import Iterator

import numpy as np
from overrides import override
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from torchvision import datasets

from crystal_property_predictor.base import (
    AugmentationFactoryBase,
    CrossValidatorFactoryBase,
    DataLoaderBase,
)
from crystal_property_predictor.dataset import CrystalDataset


class MnistDataLoader(DataLoaderBase):
    """MNIST data loading demo using DataLoaderBase."""

    @override
    def __init__(
        self,
        transforms: AugmentationFactoryBase | None,
        cross_validator: CrossValidatorFactoryBase | None,
        data_dir: str,
        batch_size: int,
        shuffle: bool,
        validation_split: float,
        nworkers: int,
        train: bool = True,
    ):
        self.data_dir = data_dir
        self.train_dataset: Dataset = datasets.MNIST(
            self.data_dir,
            train=train,
            download=True,
            transform=transforms.build_transforms(train=True),
        )
        self.valid_dataset: Dataset = (
            datasets.MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=transforms.build_transforms(train=False),
            )
            if train
            else None
        )

        # Not yet implemented for MNIST
        self.cross_validator: CrossValidatorFactoryBase | None = None

        self.init_kwargs: dict = {"batch_size": batch_size, "num_workers": nworkers}
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    @override
    def split_validation(self) -> DataLoader | None:
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)

    @override
    def generate_cross_validation_folds(
        self,
    ) -> Iterator[tuple[DataLoader, DataLoader]] | None:
        raise NotImplementedError


class CrystalDataLoader(DataLoaderBase):
    """Load crystal data."""

    @override
    def __init__(
        self,
        transforms: AugmentationFactoryBase | None,
        cross_validator: CrossValidatorFactoryBase | None,
        data_dir: str,
        batch_size: int,
        shuffle: bool,
        validation_split: float,
        nworkers: int,
        train: bool = True,
    ):
        self.data_dir = data_dir

        self.train_dataset: Dataset = CrystalDataset(
            self.data_dir,
            train=train,
            download=False,
            transform=transforms.build_transforms(train=True),
        )

        self.cross_validator: CrossValidatorFactoryBase | None = None
        self.valid_dataset: Dataset | None = None

        if train:
            self.cross_validator = cross_validator.build_validator()
            if self.cross_validator is None:
                self.train_dataset, self.valid_dataset = random_split(
                    self.train_dataset, [1 - validation_split, validation_split]
                )

        self.init_kwargs: dict = {"batch_size": batch_size, "num_workers": nworkers}
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    @override
    def split_validation(self) -> DataLoader | None:
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)

    @override
    def generate_cross_validation_folds(
        self,
    ) -> Iterator[tuple[DataLoader, DataLoader]] | None:
        if self.cross_validator is not None:
            fold: int
            train_idx: np.ndarray
            val_idx: np.ndarray
            for fold, (train_idx, val_idx) in enumerate(
                self.cross_validator.split(np.arange(len(self.train_dataset)))
            ):
                train_sampler: SubsetRandomSampler = SubsetRandomSampler(train_idx)
                valid_sampler: SubsetRandomSampler = SubsetRandomSampler(val_idx)
                train_loader: DataLoader = DataLoader(
                    self.train_dataset, sampler=train_sampler, **self.init_kwargs
                )
                valid_loader: DataLoader = DataLoader(
                    self.train_dataset, sampler=valid_sampler, **self.init_kwargs
                )

                yield train_loader, valid_loader
