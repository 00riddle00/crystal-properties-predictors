from typing import Iterator

from torch.utils.data import DataLoader


class DataLoaderBase(DataLoader):
    """Base class for all data loaders."""

    def split_validation(self) -> DataLoader | None:
        """.

        Return a `torch.utils.data.DataLoader` for validation, or None if not
        available.
        """
        raise NotImplementedError

    def generate_cross_validation_folds(
        self,
    ) -> Iterator[tuple[DataLoader, DataLoader]] | None:
        """.

        For every cross validation fold, yield a tuple of two
        `torch.utils.data.DataLoader` objects, for training and validation,
        respectively. If cross validation is not available, return None.
        """
        raise NotImplementedError
