import abc


class CrossValidatorFactoryBase(abc.ABC):
    @abc.abstractmethod
    def build_validator(self):
        pass
