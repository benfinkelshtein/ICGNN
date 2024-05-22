from torch import Tensor
from torch.nn import Module, Linear

from icgnn.classes import TransArgs


class ComLayer(Module):
    def __init__(self, in_dim: int, out_dim: int, icgnn_args: TransArgs):
        super().__init__()
        self.com_trans = icgnn_args.get_model(dim=in_dim)
        self.lin_node = Linear(in_dim, out_dim)
        self.lin_comm = Linear(in_dim, out_dim)

    def forward(self, x: Tensor, feat_mat: Tensor, affiliate_times_scale: Tensor) -> Tensor:
        community_outputs = self.com_trans(feat_mat)  # (K, C)
        out = self.lin_comm(affiliate_times_scale @ community_outputs)  # (N, C)
        out = out + self.lin_node(x)
        return out  # (N, C)

