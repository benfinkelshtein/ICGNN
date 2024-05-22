import torch
from torch import Tensor
from typing import Tuple
from torch_geometric.utils import to_dense_adj
from torch_geometric.typing import Adj
import os
import random
import numpy as np

from cutnorm import compute_cutnorm
from helpers.constants import EPS, ROOT_DIR
from icg_approximation.classes import DecompArgs, DecompTrainArgs


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def inv_sigmoid(x: Tensor) -> Tensor:
    return torch.log(x / (1 - x))


def transform_eigenvals_and_eigenvecs(eigenvals: Tensor, eigenvecs: Tensor) -> Tuple[Tensor, Tensor]:
    max_eigenvecs = torch.max(eigenvecs, dim=0)[0]
    new_eigenvecs = (eigenvecs + EPS) / (max_eigenvecs + EPS)

    new_eigenvals = eigenvals * (max_eigenvecs ** 2)
    return new_eigenvals, new_eigenvecs


def get_cut_norm(model, edge_index: Adj) -> Tuple[float, float]:
    affiliate_mat = model.affiliate_mat
    affiliate_times_scale = model.affiliate_times_scale

    model_adj = affiliate_times_scale @ affiliate_mat.T  # (N, N)
    model_adj = model_adj.cpu().detach().numpy()
    data_adj = to_dense_adj(edge_index=edge_index).squeeze(dim=0)
    data_adj = data_adj.cpu().detach().numpy()
    cutn_round, cutn_sdp, _ = compute_cutnorm(A=model_adj, B=data_adj)
    return cutn_round, cutn_sdp


def exp_path(dataset_name: str, icg_approx_args: DecompArgs, icg_approx_train_args: DecompTrainArgs, seed: int) -> str:
    run_folder = f'{dataset_name}_' \
                 f'{icg_approx_args.num_communities}_' \
                 f'Enc{int(icg_approx_args.encode_dim)}_' \
                 f'Eig{int(icg_approx_args.add_eigen)}_' \
                 f'{icg_approx_train_args.epochs}_' \
                 f'{icg_approx_train_args.lr}_' \
                 f'{icg_approx_train_args.loss_scale}' \
                 f'{seed}'
    if icg_approx_train_args.node_drop_ratio > 0:
        run_folder += f'_{icg_approx_train_args.node_drop_ratio}'
    return os.path.join(ROOT_DIR, 'icg_approximation_models', run_folder)


def get_init_com_scale_and_affiliate_mat(num_communities: int, init_com_scale: Tensor,
                                         init_affiliate_mat: Tensor) -> Tuple[Tensor, Tensor]:
    
    # com_scale
    com_scale = 2 * init_com_scale.repeat_interleave(3)[:num_communities]
    com_scale[2::3] = - com_scale[2::3] / 2

    # normalize
    pos_support = torch.relu(init_affiliate_mat)
    neg_support = torch.relu(-init_affiliate_mat)
    cross_support = pos_support + neg_support

    # affiliate_mat - sigmoid inverse
    affiliate_mat = torch.zeros(size=(init_affiliate_mat.shape[0], num_communities), device=init_com_scale.device)
    com_scale[::3], affiliate_mat[:, ::3] = \
        transform_eigenvals_and_eigenvecs(eigenvals=com_scale[::3], eigenvecs=pos_support)
    if num_communities % 3 == 0:
        com_scale[1::3], affiliate_mat[:, 1::3] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[1::3], eigenvecs=neg_support)
        com_scale[2::3], affiliate_mat[:, 2::3] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[2::3], eigenvecs=cross_support)
    elif num_communities % 3 == 1:
        com_scale[1::3], affiliate_mat[:, 1::3] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[1::3], eigenvecs=neg_support[:, :-1])
        com_scale[2::3], affiliate_mat[:, 2::3] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[2::3], eigenvecs=cross_support[:, :-1])
    elif num_communities % 3 == 2:
        com_scale[1::3], affiliate_mat[:, 1::3] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[1::3], eigenvecs=neg_support)
        com_scale[2::3], affiliate_mat[:, 2::3] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[2::3], eigenvecs=cross_support[:, :-1])
    return com_scale, affiliate_mat
