import abc

import torchvision.transforms as T
from overrides import override


class AugmentationFactoryBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class MNISTTransforms(AugmentationFactoryBase):
    MEANS = [0]
    STDS = [1]

    @override
    def build_train(self):
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])

    @override
    def build_test(self):
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])


# TODO implement selecting no augmentations in config differently than this
# This is a dummy augmentation factory that does nothing
class NoAugmentation(AugmentationFactoryBase):
    @override
    def build_train(self):
        return None

    @override
    def build_test(self):
        return None
