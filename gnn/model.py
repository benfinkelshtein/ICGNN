from torch import Tensor
from torch.nn import Module, Identity, Linear
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import Adj
import torch.nn.functional as F

from gnn.classes import GNNArgs


class GNN(Module):
    def __init__(self, model_args: GNNArgs):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        if model_args.model_type.is_gps():
            self.encoder = Linear(in_features=model_args.in_dim, out_features=model_args.hidden_dim)
            self.decoder = Linear(in_features=model_args.hidden_dim, out_features=model_args.out_dim)
        else:
            self.encoder = Identity()
            self.decoder = Identity()
        self.net = model_args.load_net()

        self.act = model_args.act_type.get()
        self.skip = model_args.skip

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = self.encoder(x)
        for idx, layer in enumerate(self.net[:-1]):
            y = layer(x=x, edge_index=edge_index)
            if self.skip and idx != 0:
                x = x + y
            else:
                x = y
            x = self.act(x)
        x = self.net[-1](x=x, edge_index=edge_index)  # (num_nodes, dim)
        x = self.decoder(x)
        return x
