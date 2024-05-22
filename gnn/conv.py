import torch
from torch import Tensor
from torch.nn import Linear

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import Adj


class GNNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, aggr: str, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(2 * in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, size=None)
        out = self.lin(torch.cat((x, out), dim=-1))
        return out
