from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from crystal_properties_predictors.base import DataLoaderBase
from crystal_properties_predictors.dataset import CrystalDataset


class MnistDataLoader(DataLoaderBase):
    """MNIST data loading demo using DataLoaderBase."""

    def __init__(
        self,
        transforms,
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

        self.init_kwargs = {"batch_size": batch_size, "num_workers": nworkers}
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)


class CrystalDataLoader(DataLoaderBase):
    """Load crystal data."""

    def __init__(
        self,
        transforms,
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

        if train:
            self.train_dataset, self.valid_dataset = random_split(
                self.train_dataset, [1 - validation_split, validation_split]
            )
        else:
            self.valid_dataset = None

        self.init_kwargs = {"batch_size": batch_size, "num_workers": nworkers}
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)
