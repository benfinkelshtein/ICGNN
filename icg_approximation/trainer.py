import os.path
import torch
from torch_geometric.data import Data
import sys
import tqdm
import os.path as osp

from icg_approximation.classes import DecompTrainArgs
from icg_approximation.model import DecompModel
from icg_approximation.utils import set_seed
from icg_approximation.train import train_decomp
from helpers.constants import DECIMAL


class DecompTrainer(object):

    def __init__(self, train_args: DecompTrainArgs, seed: int, device, exp_path: str):
        super().__init__()
        self.train_args = train_args
        self.seed = seed
        self.device = device
        self.exp_path = exp_path

    def train(self, model: DecompModel, data: Data, is_spatio_temporal: bool):
        set_seed(seed=self.seed)
        icg_approx_optimizer = torch.optim.Adam(model.parameters(), lr=self.train_args.lr)
        with tqdm.tqdm(total=self.train_args.epochs, file=sys.stdout) as pbar:
            best_loss, best_grahon_re, model = \
                train_decomp(data=data, model=model, optimizer=icg_approx_optimizer, train_args=self.train_args,
                             pbar=pbar, device=self.device, is_spatio_temporal=is_spatio_temporal)

        # Save
        print('Saving Decomp Model')
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        delattr(model, 'feat_mat')
        torch.save(model.state_dict(), osp.join(self.exp_path, 'model.pt'))
        torch.save((best_loss, best_grahon_re), osp.join(self.exp_path, 'loss.pt'))

        # print
        print(f'Final GS;loss={round(best_loss, DECIMAL)};g_re={round(best_grahon_re, DECIMAL)}')
        return model
