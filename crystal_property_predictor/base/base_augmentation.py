import abc
from typing import Callable


class AugmentationFactoryBase(abc.ABC):
    def build_transforms(self, train: bool) -> Callable | None:
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self) -> Callable | None:
        pass

    @abc.abstractmethod
    def build_test(self) -> Callable | None:
        pass
