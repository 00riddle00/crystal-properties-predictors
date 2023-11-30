from typing import Callable

import torchvision.transforms as T
from overrides import override

from crystal_property_predictor.base import AugmentationFactoryBase


class MNISTTransforms(AugmentationFactoryBase):
    MEANS: list[int] = [0]
    STDS: list[int] = [1]

    @override
    def build_train(self) -> Callable | None:
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])

    @override
    def build_test(self) -> Callable | None:
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])


# TODO implement selecting no augmentations in config differently than this
# This is a dummy augmentation factory that does nothing
class NoAugmentation(AugmentationFactoryBase):
    @override
    def build_train(self) -> Callable | None:
        return None

    @override
    def build_test(self) -> Callable | None:
        return None
