from typing import Tuple

from torch.utils.data import DataLoader


class DataLoaderBase(DataLoader):
    """Base class for all data loaders."""

    def split_validation(self) -> DataLoader:
        """.

        Return a `torch.utils.data.DataLoader` for validation, or None if not
        available.
        """
        raise NotImplementedError

    def generate_cross_validation_folds(self) -> Tuple[DataLoader, DataLoader]:
        """.

        For every cross validation fold, yield a tuple of two
        `torch.utils.data.DataLoader` objects, for training and validation.
        """
        raise NotImplementedError
