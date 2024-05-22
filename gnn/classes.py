from torch.nn import ModuleList
from typing import NamedTuple

from enum import Enum, auto
from torch_geometric.nn.conv import GATConv, GPSConv, GCNConv
from torch.nn import Module

from gnn.conv import GNNConv
from additional_classes.activation import ActivationType


class ModelType(Enum):
    """
        an object for the different core
    """
    MEAN_GNN = auto()
    GCN = auto()
    GAT = auto()
    GPS = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()

    def load_component(self, in_channels: int, out_channels: int) -> Module:
        if self is ModelType.MEAN_GNN:
            return GNNConv(in_channels=in_channels, out_channels=out_channels, aggr='mean')
        elif self is ModelType.GAT:
            return GATConv(in_channels=in_channels, out_channels=out_channels)
        elif self is ModelType.GCN:
            return GCNConv(in_channels=in_channels, out_channels=out_channels)
        elif self is ModelType.GPS:
            mean_gnn = GNNConv(in_channels=in_channels, out_channels=in_channels, aggr='mean')
            return GPSConv(channels=in_channels, conv=mean_gnn, norm=None)
        else:
            raise ValueError(f'model {self.name} not supported')

    def is_gps(self) -> bool:
        return self is ModelType.GPS


class GNNArgs(NamedTuple):
    model_type: ModelType
    num_layers: int
    in_dim: int
    hidden_dim: int
    out_dim: int
    act_type: ActivationType
    skip: bool

    def load_net(self) -> ModuleList:
        if self.model_type.is_gps():
            dim_list = [self.hidden_dim] * (self.num_layers + 1)
        else:
            dim_list = [self.in_dim] + [self.hidden_dim] * (self.num_layers - 1) + [self.out_dim]
        component_list = [self.model_type.load_component(in_channels=in_dim_i, out_channels=out_dim_i)
                          for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        return ModuleList(component_list)
