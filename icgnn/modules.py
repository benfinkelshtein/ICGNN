from torch.nn import Parameter, Module, init, Linear, ModuleList, Dropout, MultiheadAttention
from torch import Tensor
import torch
import math

from additional_classes.activation import ActivationType


class MHA(Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float, skip: bool,
                 act_type: ActivationType):
        super().__init__()
        self.add_encoder = in_dim != hidden_dim
        list_of_modules = []
        if self.add_encoder:
            list_of_modules += [Linear(in_features=in_dim, out_features=hidden_dim)]
        list_of_modules += [MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True, dropout=dropout)
                            for _ in range(num_layers)]
        list_of_modules += [Linear(in_features=hidden_dim, out_features=out_dim)]
        self.module_list = ModuleList(list_of_modules)
        self.dropout = Dropout(dropout)

        self.act = act_type.get()
        self.skip = skip

    def forward(self, x: Tensor) -> Tensor:
        start = 0
        if self.add_encoder:
            start = 1

            x = self.module_list[0](x)
            x = self.act(x)
            x = self.dropout(x)

        for layer in self.module_list[start:-1]:
            y, _ = layer(x, x, x)
            y = self.act(y)
            if self.skip:
                x = x + y
            else:
                x = y

        return self.module_list[-1](x)


class MAT(Module):
    def __init__(self, num_communities: int, out_dim: int):
        super().__init__()
        self.num_communities = num_communities
        self.mat = Parameter(torch.zeros((num_communities, out_dim)))  # (K, C)
        self.reset_parameters()

    def reset_parameters(self):
        # feat_mat is inspired by torch.nn.modules.linear class Linear (as both are linear transformations)
        bound = 1 / math.sqrt(self.num_communities)
        init.uniform_(self.mat, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return self.mat
