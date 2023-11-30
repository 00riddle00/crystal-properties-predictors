import abc
from typing import Callable


class CrossValidatorFactoryBase(abc.ABC):
    @abc.abstractmethod
    def build_validator(self) -> Callable | None:
        pass
