from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv.simple_conv import MessagePassing


class FrobeniusNormMessagePassing(MessagePassing):
    def __init__(self, com_scale: Tensor):
        super().__init__(aggr='sum')
        self.com_scale = com_scale  # (K,)

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        # x_i is the target node of dim (E, K)
        # x_j is the source node of dim (E, K)
        message_term = (x_i * self.com_scale * x_j).sum(dim=1, keepdims=True)  # (E, )
        if edge_weight is not None:
            message_term = message_term * edge_weight.unsqueeze(dim=1)
        return message_term

    def forward(self, affiliate_mat: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        node_based_result = self.propagate(edge_index, x=affiliate_mat, edge_weight=edge_weight)  # (N, 1)
        return node_based_result.squeeze(dim=1).sum(dim=0)  # (,)
