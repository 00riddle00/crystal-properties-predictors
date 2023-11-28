import numpy as np
from overrides import override
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torchvision import datasets

from crystal_property_predictor.base import DataLoaderBase
from crystal_property_predictor.dataset import CrystalDataset


class MnistDataLoader(DataLoaderBase):
    """MNIST data loading demo using DataLoaderBase."""

    @override
    def __init__(
        self,
        transforms,
        cross_validator,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        nworkers,
        train=True,
    ):
        self.data_dir = data_dir

        self.train_dataset = datasets.MNIST(
            self.data_dir,
            train=train,
            download=True,
            transform=transforms.build_transforms(train=True),
        )
        self.valid_dataset = (
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
        self.cross_validator = None

        self.init_kwargs = {"batch_size": batch_size, "num_workers": nworkers}
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    @override
    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)

    @override
    def generate_cross_validation_folds(self):
        raise NotImplementedError


class CrystalDataLoader(DataLoaderBase):
    """Load crystal data."""

    @override
    def __init__(
        self,
        transforms,
        cross_validator,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        nworkers,
        train=True,
    ):
        self.data_dir = data_dir

        self.train_dataset = CrystalDataset(
            self.data_dir,
            train=train,
            download=False,
            transform=transforms.build_transforms(train=True),
        )

        self.cross_validator = None
        self.valid_dataset = None

        if train:
            self.cross_validator = cross_validator.build_validator()
            if self.cross_validator is None:
                self.train_dataset, self.valid_dataset = random_split(
                    self.train_dataset, [1 - validation_split, validation_split]
                )

        self.init_kwargs = {"batch_size": batch_size, "num_workers": nworkers}
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    @override
    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)

    @override
    def generate_cross_validation_folds(self):
        if self.cross_validator is not None:
            for fold, (train_idx, val_idx) in enumerate(
                self.cross_validator.split(np.arange(len(self.train_dataset)))
            ):
                train_sampler = SubsetRandomSampler(train_idx)
                valid_sampler = SubsetRandomSampler(val_idx)
                train_loader = DataLoader(
                    self.dataset, sampler=train_sampler, **self.init_kwargs
                )
                valid_loader = DataLoader(
                    self.dataset, sampler=valid_sampler, **self.init_kwargs
                )

                yield train_loader, valid_loader
