import os
from typing import Callable, Tuple

import pandas as pd
import torch
from overrides import override

from crystal_property_predictor.base import DatasetBase


class CrystalDataset(DatasetBase):
    """Crystals Dataset.

    Args:
        root (string): Root directory of dataset where
            ``CrystalDataset/raw/train-crystal-structures.csv`` and
            ``CrystalDataset/test-crystal-structures.csv`` exist.
        train (bool, optional): If True, creates dataset from
           ``train-crystal-structures.csv``,
            otherwise from ``test-crystal-structures.csv``.
        transform (callable, optional): A function/transform that takes in the
            feature vector and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
    """

    resources = [
        ("train-crystal-structures.csv", "<place_for_md5_hash>"),
        ("train-crystal-melting-points.csv", "<place_for_md5_hash>"),
        # ("test-crystal-structures.csv", "<place_for_md5_hash>"),
        # ("test-crystal-melting-points.csv", "<place_for_md5_hash>"),
    ]

    @override
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root

        # TODO set self.transform and self.target_transform according to
        #  `transform` and `target_transform` arguments (see
        #  torchvision.datasets.VisionDataset)
        pass

        if download:
            self.download()

        # training set or test set
        self.train = train

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets = self._load_data()

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        structure_file: str = (
            f"{'train' if self.train else 'test'}-crystal-structures.csv"
        )

        melting_point_file: str = (
            f"{'train' if self.train else 'test'}-crystal-melting-points.csv"
        )

        data_df: pd.DataFrame = (
            pd.read_csv(os.path.join(self.raw_folder, structure_file))
            .set_index(["CODID"])
            .sort_index()
        )

        targets_df: pd.DataFrame = (
            pd.read_csv(os.path.join(self.raw_folder, melting_point_file))
            .set_index(["CODID"])
            .sort_index()
        )

        if not self._validate_keys(data_df, targets_df):
            raise RuntimeError(
                "CODID keys in data file do not correspond to the keys in targets file."
                " Make sure both files contain the same keys."
            )

        data: torch.Tensor = torch.tensor(data_df.to_numpy(), dtype=torch.float32)
        targets: torch.Tensor = torch.tensor(targets_df.to_numpy(), dtype=torch.float32)

        return data, targets

    @staticmethod
    def _validate_keys(data: pd.DataFrame, targets: pd.DataFrame) -> bool:
        return data.index.equals(targets.index)

    # Implement if needed
    def download(self) -> None:
        """Download the crystal data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        pass

    @override
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # E.g. element count, volume, weight, etc.
        features: torch.Tensor = self.data[index].clone().detach()
        # True value, e.g. melting point
        target: torch.Tensor = self.targets[index].clone().detach()

        # TODO apply transformations according to self.transform and
        #  self.target_transform
        pass

        return features, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self) -> bool:
        return all(
            os.path.isfile((os.path.join(self.raw_folder, f_name)))
            for f_name, _ in self.resources
        )
