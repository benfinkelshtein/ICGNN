from torch import Tensor
from torch.nn import Module, ModuleList, Dropout
from torch_geometric.typing import Adj

from icgnn.classes import CommArgs, TransArgs
from icgnn.layer import ComLayer
from icg_approximation.model import DecompModel


class TransModel(Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, icgnn_args: TransArgs):
        super().__init__()
        dim_list = [in_dim] + [hidden_dim] * (icgnn_args.num_layers - 1) + [out_dim]
        self.layers = ModuleList([ComLayer(in_dim=in_channels, out_dim=out_channels, icgnn_args=icgnn_args)
                                  for in_channels, out_channels in zip(dim_list[:-1], dim_list[1:])])

        self.dropout = Dropout(icgnn_args.dropout)
        self.act = icgnn_args.act_type.get()

    def forward(self, x: Tensor, affiliate_times_scale: Tensor, inv_affiliate_mat: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            feat_mat = inv_affiliate_mat @ x  # (K, C)
            x = layer(x=x, feat_mat=feat_mat, affiliate_times_scale=affiliate_times_scale)  # (N, C)
            x = self.act(x)
            x = self.dropout(x)

        feat_mat = inv_affiliate_mat @ x  # (K, C)
        x = self.layers[-1](x=x, feat_mat=feat_mat, affiliate_times_scale=affiliate_times_scale)
        return x


class CommModel(Module):
    def __init__(self, model_args: CommArgs, icg_approx_model: DecompModel):
        super().__init__()
        self.decomp = icg_approx_model
        self.trans = TransModel(in_dim=model_args.encoded_dim, hidden_dim=model_args.hidden_dim,
                                out_dim=model_args.out_dim, icgnn_args=model_args.icgnn_args)

    # gradients
    def train(self, mode: bool = True):
        self.decomp.eval()
        self.trans.train()

    def get_icgnn_parameters(self):
        return self.trans.parameters()

    def set_icg_approx_after_training(self):
        self.decomp.set_matrices_after_icg_approx_training()
        self.decomp.requires_grad_(requires_grad=False)

    # forward pass
    def encode(self, x: Tensor) -> Tensor:
        return self.decomp.encode(x=x)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        out = self.trans(x=self.encode(x),
                         affiliate_times_scale=self.decomp.affiliate_times_scale,
                         inv_affiliate_mat=self.decomp.inv_affiliate_mat)
        return out


