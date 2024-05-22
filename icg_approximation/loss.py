import torch
from torch import Tensor
from typing import Tuple
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.data import Data

from icg_approximation.forbenius_norm_mp import FrobeniusNormMessagePassing


def drops_nodes(x: Tensor, edge_index: Adj, edge_weight: OptTensor, affiliate_mat: Tensor,
                node_drop_ratio: float) -> Tuple[Tensor, Adj, Tensor, Tensor]:
    # filter nodes
    device = x.device
    prob_vector = (1 - node_drop_ratio) * torch.ones(size=(x.shape[0],), device=device)
    indices_kept = torch.bernoulli(prob_vector).type(torch.bool)
    x = x[indices_kept]
    affiliate_mat = affiliate_mat[indices_kept]

    # filter edges
    u, v = edge_index
    u_kept, v_kept = indices_kept[u], indices_kept[v]
    edge_kept = torch.logical_and(u_kept, v_kept)
    edge_index = edge_index[:, edge_kept]
    if edge_weight is not None:
        edge_weight = edge_weight[edge_kept]

    # edge index reindexing
    _, edge_index = edge_index.unique(return_inverse=True)
    edge_index = edge_index.view(2, -1)
    return x, edge_index, edge_weight, affiliate_mat


def calc_efficient_graphon_loss(data: Data, model, loss_scale: float, device, node_drop_ratio: float,
                                is_spatio_temporal: bool) -> Tuple[Tensor, Tensor, Tensor]:
    # pre drop
    if is_spatio_temporal:
        num_time_steps, num_feat, num_nodes = data.x.shape[0], data.x.shape[1], data.x.shape[2]
    else:
        num_nodes, num_feat = data.x.shape[0], data.x.shape[1]

    # global term
    x = model.encode(x=data.x.to(device))
    com_scale = model.com_scale
    affiliate_mat = model.affiliate_mat
    global_term = torch.trace(affiliate_mat.T @ model.affiliate_times_scale @ affiliate_mat.T \
                              @ (affiliate_mat * com_scale))  # (,)

    # node drop
    edge_index = data.edge_index.to(device)
    edge_weight = torch.ones_like(edge_index[0], dtype=torch.float).to(device)
    if hasattr(data, 'edge_weight'):
        if data.edge_weight is not None:
            edge_weight = data.edge_weight.to(device)
    # NOTE: when dropping nodes the loss decreases
    if node_drop_ratio > 0:
        x, edge_index, edge_weight, affiliate_mat = \
            drops_nodes(x=x, edge_index=edge_index, edge_weight=edge_weight,
                        affiliate_mat=affiliate_mat, node_drop_ratio=node_drop_ratio)
    num_edges = (edge_weight ** 2).sum()

    frobenius_norm_mp = FrobeniusNormMessagePassing(com_scale=com_scale).to(device=device)
    local_term = 2 * frobenius_norm_mp(affiliate_mat=affiliate_mat, edge_index=edge_index, edge_weight=edge_weight)  # (,)
    data_term = num_edges
    graphon_loss = (global_term - local_term + data_term) / num_nodes

    if loss_scale > 0:
        if data.x.dim() >= 3:
            x_approx = torch.matmul((affiliate_mat * model.com_scale), model.feat_mat).unsqueeze(dim=0)
            signal_loss = torch.sum((x - x_approx) ** 2) / (num_feat * num_time_steps)
        else:
            signal_loss = torch.sum((x - (affiliate_mat * model.com_scale) @ model.feat_mat) ** 2) / num_feat
    else:
        signal_loss = 0
    graphon_re_denom = num_edges / num_nodes
    return graphon_loss + loss_scale * signal_loss, graphon_loss,  (graphon_loss / graphon_re_denom) ** 0.5
