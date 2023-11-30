from typing import Callable

from overrides import override
from sklearn.model_selection import KFold

from crystal_property_predictor.base import CrossValidatorFactoryBase


class KFoldCV(CrossValidatorFactoryBase):
    """K-fold cross validator."""

    def __init__(self, n_folds: int, shuffle: bool, random_state: int):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    @override
    def build_validator(self) -> Callable | None:
        return KFold(
            n_splits=self.n_folds,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )


# TODO implement selecting no cross validation in config differently than this
# This is a dummy cross validator that does nothing
class NoCrossValidation(CrossValidatorFactoryBase):
    @override
    def build_validator(self) -> Callable | None:
        return None
