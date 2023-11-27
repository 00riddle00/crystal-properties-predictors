import abc

from sklearn.model_selection import KFold


class CrossValidatorFactoryBase(abc.ABC):
    @abc.abstractmethod
    def build_validator(self):
        pass


class KFoldCV(CrossValidatorFactoryBase):
    """K-fold cross validator."""

    def __init__(self, n_folds, shuffle, random_state):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def build_validator(self):
        return KFold(
            n_splits=self.n_folds,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )


# TODO implement selecting no cross validation in config differently than this
# This is a dummy cross validator that does nothing
class NONE(CrossValidatorFactoryBase):
    def build_validator(self):
        pass
