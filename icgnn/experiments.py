from argparse import Namespace
import torch
import os.path as osp

from icg_approximation.classes import DecompArgs, DecompTrainArgs
from icg_approximation.utils import set_seed, exp_path
from helpers.constants import DECIMAL, RECORD_PREFIX_STR, TIME_STEPS
from icgnn.classes import TransArgs, TransTrainArgs, CommArgs
from icgnn.trainer import TransTrainer
from icg_approximation.model import DecompModel


class TransExperiment(object):
    def __init__(self, args: Namespace):
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            print(f"{arg}: {value_arg}")
            self.__setattr__(arg, value_arg)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(seed=0)

        self.metric_type = self.dataset_type.get_metric_type()
        self.task_loss = self.metric_type.get_task_loss()
        self.icg_approx_train_args = DecompTrainArgs(epochs=self.icg_approx_epochs, lr=self.icg_approx_lr,
                                                     loss_scale=self.loss_scale, cut_norm=None,
                                                     node_drop_ratio=self.node_drop_ratio)
        assert not self.dataset_type.is_spatio_temporal(), \
            "To run spatio_temporal datasets use icgnn_spatio_temporal/run_static_Graph.py"

    def run(self):
        # load data
        data = self.dataset_type.load(trans=True)[0]
        act_type = self.dataset_type.activation_type()

        # load icg_approximation args
        out_dim = self.metric_type.get_out_dim(data=data)
        time_steps = TIME_STEPS if self.dataset_type.is_spatio_temporal() else 1
        num_nodes, in_dim = data.x.shape[-2], data.x.shape[-1]
        icg_approx_model_args = DecompArgs(num_communities=self.num_communities, encode_dim=self.encode_dim,
                                           dropout=self.icg_approx_dropout, num_nodes=num_nodes, in_dim=in_dim,
                                           add_eigen=self.add_eigen, init_affiliate_mat=None,
                                           init_com_scale=None, init_feat_mat=None, time_steps=time_steps)
        path = exp_path(dataset_name=self.dataset_type.name, icg_approx_args=icg_approx_model_args,
                        icg_approx_train_args=self.icg_approx_train_args, seed=0)

        # load icg_approximation model
        print('Loading Decomp Model')
        icg_approx_model = DecompModel(model_args=icg_approx_model_args, trans=True).to(self.device)
        state_dict = torch.load(osp.join(path, 'model.pt'))
        if 'feat_mat' in state_dict:
            del state_dict['feat_mat']
        if hasattr(icg_approx_model, 'feat_mat'):
            delattr(icg_approx_model, 'feat_mat')
        icg_approx_model.load_state_dict(state_dict)
        best_loss, best_grahon_re = torch.load(osp.join(path, 'loss.pt'))
        print(f'Final GS;loss={round(best_loss, DECIMAL)};g_re={round(best_grahon_re, DECIMAL)}')

        # load icgnn args
        icgnn_args = TransArgs(icgnn_type=self.icgnn_type, nn_num_layers=self.nn_num_layers,
                               dropout=self.dropout, skip=self.skip, act_type=act_type,
                               num_layers=self.num_layers, num_communities=self.num_communities)
        model_args = CommArgs(encoded_dim=icg_approx_model_args.encoded_dim, hidden_dim=self.hidden_dim,
                              out_dim=out_dim,
                              icgnn_args=icgnn_args)
        train_args = TransTrainArgs(epochs=self.epochs, lr=self.lr)

        # train
        trainer = TransTrainer(model_args=model_args, train_args=train_args, seed=0, device=self.device,
                               is_gnn=False)
        trainer.train_and_test_splits(dataset_type=self.dataset_type, data=data, icg_approx_model=icg_approx_model)
