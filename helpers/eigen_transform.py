from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)
import scipy

# Similar to torch_geometric.transforms.add_positional_encoding


def add_node_attr(
    data: Data,
    value: Any,
    attr_name: Optional[str] = None,
) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


class AddLargestAbsEigenvecAndEigenval(BaseTransform):
    def __init__(
        self,
        k: int,
        is_undirected: bool = False,
        **kwargs: Any,
    ) -> None:
        self.k = k
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        if data.x.dim() >= 3:  # spatio-temporal
            num_nodes = data.num_nodes
        else:
            num_nodes = data.x.shape[0]
        assert num_nodes is not None
        if self.k > num_nodes:
            self.k = num_nodes

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
        if self.k > num_nodes * 0.8:
            eig_vals, eig_vecs = scipy.linalg.eigh(adj.todense())
        else:
            eig_vals, eig_vecs = scipy.sparse.linalg.eigsh(  # additional_classes: ignore
                adj,
                k=self.k,
                which='LM',  # largest magnitude
                return_eigenvectors=True,
                **self.kwargs,
            )

        # changes
        largest_indices = np.abs(eig_vals).argsort()[::-1]
        eig_vals = np.real(eig_vals[largest_indices])
        eig_vecs = np.real(eig_vecs[:, largest_indices])

        eig_vals = torch.from_numpy(eig_vals[:self.k])
        pe = torch.from_numpy(eig_vecs[:, :self.k])

        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        pe *= sign

        data = add_node_attr(data, pe, attr_name='eigenvecs')
        data = add_node_attr(data, eig_vals, attr_name='eigenvals')
        return data
