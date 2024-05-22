from argparse import Namespace
import torch
import math

from icg_approximation.classes import DecompArgs, DecompTrainArgs
from icg_approximation.utils import set_seed, exp_path, get_init_com_scale_and_affiliate_mat
from helpers.constants import TIME_STEPS
from icg_approximation.trainer import DecompTrainer
from icg_approximation.model import DecompModel


class DecompExperiment(object):
    def __init__(self, args: Namespace):
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            print(f"{arg}: {value_arg}")
            self.__setattr__(arg, value_arg)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(seed=0)

        self.train_args = DecompTrainArgs(epochs=self.icg_approx_epochs, lr=self.icg_approx_lr, loss_scale=self.loss_scale,
                                          cut_norm=self.cut_norm, node_drop_ratio=self.node_drop_ratio)
        assert not(self.dataset_type.is_spatio_temporal() and self.node_drop_ratio > 0),\
            f'Spatio-temporal datasets do not work with node_drop_ratio'

    def run(self):
        # load data
        data = self.dataset_type.load(trans=False)[0]
        delattr(data, 'train_mask')
        delattr(data, 'val_mask')
        delattr(data, 'test_mask')

        # initialize model
        init_com_scale, init_affiliate_mat, init_feat_mat = None, None, None
        if self.add_eigen:
            assert hasattr(data, 'eigenvecs'), f'No PE found, re-download the data!'
            init_com_scale = data.eigenvals[:math.ceil(self.num_communities / 3)]
            init_affiliate_mat = data.eigenvecs[:, :math.ceil(self.num_communities / 3)]
            full_com_scale, full_affiliate_mat =\
                get_init_com_scale_and_affiliate_mat(num_communities=self.num_communities,
                                                     init_com_scale=init_com_scale,
                                                     init_affiliate_mat=init_affiliate_mat)
            if self.encode_dim == 0:
                if self.dataset_type.is_spatio_temporal():
                    init_feat_mat = torch.linalg.pinv(full_affiliate_mat * full_com_scale).unsqueeze(dim=0) @ data.x
                else:
                    init_feat_mat = torch.linalg.pinv(full_affiliate_mat * full_com_scale) @ data.x
        delattr(data, 'eigenvals')
        delattr(data, 'eigenvecs')

        # load args
        time_steps = TIME_STEPS if self.dataset_type.is_spatio_temporal() else 1
        num_nodes, in_dim = data.x.shape[-2], data.x.shape[-1]
        model_args = DecompArgs(num_communities=self.num_communities, encode_dim=self.encode_dim,
                                num_nodes=num_nodes, in_dim=in_dim, add_eigen=self.add_eigen,
                                node_drop_ratio=self.node_drop_ratio,
                                init_affiliate_mat=init_affiliate_mat, init_com_scale=init_com_scale,
                                init_feat_mat=init_feat_mat, time_steps=time_steps)
        model = DecompModel(model_args=model_args, trans=False).to(device=self.device)

        # train
        path = exp_path(dataset_name=self.dataset_type.name, icg_approx_args=model_args, icg_approx_train_args=self.train_args,
                        seed=0)
        trainer = DecompTrainer(train_args=self.train_args, seed=0,
                                device=self.device, exp_path=path)
        model = trainer.train(model=model, data=data, is_spatio_temporal=self.dataset_type.is_spatio_temporal())
