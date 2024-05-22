from typing import NamedTuple, Optional
from torch_geometric.typing import OptTensor


class DecompTrainArgs(NamedTuple):
    epochs: int
    lr: float
    loss_scale: float
    cut_norm: Optional[bool]
    node_drop_ratio: float


class DecompArgs(NamedTuple):
    num_communities: int
    encode_dim: int

    num_nodes: int
    in_dim: int

    add_eigen: bool
    node_drop_ratio: float
    time_steps: int
    init_affiliate_mat: OptTensor
    init_com_scale: OptTensor
    init_feat_mat: OptTensor

    @property
    def encoded_dim(self) -> int:
        return self.in_dim if self.encode_dim == 0 else self.encode_dim
