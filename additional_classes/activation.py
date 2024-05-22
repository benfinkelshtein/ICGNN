from enum import Enum, auto
from torch.nn import Module, ReLU, GELU


class ActivationType(Enum):
    """
        an object for the different activation types
    """
    RELU = auto()
    GELU = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ActivationType[s]
        except KeyError:
            raise ValueError()

    def get(self) -> Module:
        if self is ActivationType.RELU:
            return ReLU()
        elif self is ActivationType.GELU:
            return GELU()
        else:
            raise ValueError(f'ActivationType {self.name} not supported')