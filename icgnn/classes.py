from typing import NamedTuple, Optional
from torch.nn import Module
from enum import Enum, auto

from additional_classes.activation import ActivationType
from icgnn.modules import MHA, MAT


class TransType(Enum):
    """
        an object for the different activation types
    """
    Matrix = auto()
    MHA = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return TransType[s]
        except KeyError:
            raise ValueError()

    def is_mat(self) -> bool:
        return self is TransType.Matrix


class TransArgs(NamedTuple):
    num_layers: int
    icgnn_type: TransType
    nn_num_layers: int
    dropout: float
    skip: bool
    act_type: ActivationType

    num_communities: int

    def get_model(self, dim: int) -> Optional[Module]:
        if self.icgnn_type is TransType.MLP:
            return MLP(in_dim=dim, hidden_dim=dim, out_dim=dim,
                       num_layers=self.nn_num_layers, dropout=self.dropout,
                       skip=self.skip, act_type=self.act_type, num_communities=self.num_communities)
        elif self.icgnn_type is TransType.Matrix:
            return MAT(num_communities=self.num_communities, out_dim=dim)
        elif self.icgnn_type is TransType.MHA:
            return MHA(in_dim=dim, hidden_dim=dim, out_dim=dim,
                               num_layers=self.nn_num_layers, dropout=self.dropout,
                               skip=self.skip, act_type=self.act_type)
        else:
            raise ValueError(f'TransType {self.icgnn_type.name} not supported')


class CommArgs(NamedTuple):
    encoded_dim: int
    hidden_dim: int
    out_dim: int
    icgnn_args: TransArgs


class TransTrainArgs(NamedTuple):
    epochs: int
    lr: float

