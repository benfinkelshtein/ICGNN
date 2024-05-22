from argparse import Namespace
import torch
from torch_geometric.transforms import AddRandomWalkPE

from icg_approximation.utils import set_seed
from icgnn.trainer import TransTrainer
from icgnn.classes import TransTrainArgs
from gnn.classes import GNNArgs


class GNNExperiment(object):
    def __init__(self, args: Namespace):
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            print(f"{arg}: {value_arg}")
            self.__setattr__(arg, value_arg)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(seed=0)

        self.icgnn_train_args = TransTrainArgs(epochs=self.epochs, lr=self.lr)
        self.metric_type = self.dataset_type.get_metric_type()
        self.task_loss = self.metric_type.get_task_loss()
        assert not self.dataset_type.is_spatio_temporal(), "GNN doesn\'t work for spatio-temporal datasets"

    def run(self):
        # load data
        pos_enc_transform = None
        if self.model_type.is_gps() and self.rw_pos_length > 0:
            pos_enc_transform = AddRandomWalkPE(walk_length=self.rw_pos_length)
        data = self.dataset_type.load(trans=False, pos_enc_transform=pos_enc_transform)[0]

        # load args
        in_dim = data.x.shape[1]
        out_dim = self.metric_type.get_out_dim(data=data)
        act_type = self.dataset_type.activation_type()
        model_args = GNNArgs(model_type=self.model_type, num_layers=self.num_layers,
                             in_dim=in_dim + self.rw_pos_length,
                             hidden_dim=self.hidden_dim, out_dim=out_dim, act_type=act_type, skip=self.skip)

        # train
        standard_trainer = TransTrainer(model_args=model_args, train_args=self.icgnn_train_args,
                                        seed=0, device=self.device, is_gnn=True)
        standard_trainer.train_and_test_splits(dataset_type=self.dataset_type, data=data, icg_approx_model=None)
