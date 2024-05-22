import torch
from torch import Tensor
from torch.nn import Parameter, Module, init, Linear, Identity
import math
from torch_geometric.typing import OptTensor

from icg_approximation.classes import DecompArgs
from icg_approximation.utils import get_init_com_scale_and_affiliate_mat, inv_sigmoid
from icg_approximation.scale_gradient import ScaleGradientModule


class DecompModel(Module):
    def __init__(self, model_args: DecompArgs, trans: bool):
        super().__init__()
        num_nodes = model_args.num_nodes
        num_communities = model_args.num_communities
        self.num_communities = num_communities

        # nodes
        self._affiliate_mat = Parameter(torch.zeros((num_nodes, num_communities)))
        self.com_scale = Parameter(torch.zeros((num_communities,)))
        self._inv_affiliate_mat = None
        
        # features
        self.encoder = None
        if model_args.encode_dim > 0:
            self.encoder = Linear(in_features=model_args.in_dim, out_features=model_args.encode_dim)
        if model_args.time_steps > 1:
            self.feat_mat = Parameter(torch.zeros((model_args.time_steps, num_communities, model_args.encoded_dim)))
        else:
            self.feat_mat = Parameter(torch.zeros((num_communities, model_args.encoded_dim)))

        # node drop
        if model_args.node_drop_ratio > 0:
            num_left = int(num_nodes * (1 - model_args.node_drop_ratio))
            self.scale_grad = ScaleGradientModule(scale=num_left / num_nodes)
        else:
            self.scale_grad = Identity()

        if not trans:
            self.reset_parameters(init_com_scale=model_args.init_com_scale,
                                  init_affiliate_mat=model_args.init_affiliate_mat,
                                  init_feat_mat=model_args.init_feat_mat, )

    def reset_parameters(self, init_com_scale: OptTensor, init_affiliate_mat: OptTensor, init_feat_mat: OptTensor):
        if init_com_scale is None:
            # feat_mat is inspired by torch.nn.modules.linear class Linear (as both are linear transformations)
            bound = 1 / math.sqrt(self.num_communities)
            init.uniform_(self.feat_mat, -bound, bound)

            # com_scale is inspired by the bias of torch.nn.modules.linear class Linear
            init.uniform_(self.com_scale, -bound, bound)

            # _affiliate_mat
            init.uniform_(self._affiliate_mat, -4, 4)
        else:
            self.com_scale.data, self._affiliate_mat.data =\
                get_init_com_scale_and_affiliate_mat(num_communities=self.num_communities,
                                                     init_com_scale=init_com_scale,
                                                     init_affiliate_mat=init_affiliate_mat)
            self._affiliate_mat.data = inv_sigmoid(x=self._affiliate_mat.data)
        if init_feat_mat is None:
            # feat_mat is inspired by torch.nn.modules.linear class Linear (as both are linear transformations)
            bound = 1 / math.sqrt(self.num_communities)
            init.uniform_(self.feat_mat, -bound, bound)
        else:
            self.feat_mat.data = init_feat_mat
        
        if self.encoder is not None:
            self.encoder.reset_parameters()

    @property
    def affiliate_mat(self) -> Tensor:
        return torch.sigmoid(self.scale_grad(self._affiliate_mat))
    
    @property
    def affiliate_times_scale(self) -> Tensor:
        return self.affiliate_mat * self.com_scale

    def set_matrices_after_icg_approx_training(self):
        self._inv_affiliate_mat = torch.linalg.pinv(self.affiliate_times_scale).detach()

    @property
    def inv_affiliate_mat(self) -> Tensor:
        assert self._inv_affiliate_mat is not None,\
            "Call set_matrices_after_icg_approx_training before using inv_affiliate_mat"
        return self._inv_affiliate_mat.to(self.com_scale.device)

    def encode(self, x: Tensor) -> Tensor:
        if self.encoder is not None:
            return self.encoder(x)
        else:
            return x
