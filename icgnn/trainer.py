import torch
from typing import Tuple, Union, Optional
from torch_geometric.data import Data
from torch.optim import Optimizer
import sys
import tqdm

from additional_classes.dataset import DataSet
from additional_classes.metrics import MetricType, LossesAndMetrics
from icgnn.classes import TransTrainArgs, CommArgs
from icg_approximation.model import DecompModel
from icg_approximation.utils import set_seed
from icgnn.utils import icgnn_final_log_and_print, epoch_log_and_print
from icgnn.model import CommModel
from helpers.constants import DECIMAL


class TransTrainer(object):
    def __init__(self, model_args: CommArgs, train_args: TransTrainArgs,
                 seed: int, device):
        super().__init__()
        self.model_args = model_args
        self.train_args = train_args
        self.seed = seed
        self.device = device

    def train_and_test_splits(self, dataset_type: DataSet, data: Data,
                              icg_approx_model: Optional[DecompModel]) -> CommModel:
        folds = dataset_type.get_folds()
        metrics_list = []
        for num_fold in folds:
            set_seed(seed=self.seed)
            data_split = dataset_type.select_fold(num_fold=num_fold, data=data)
            print_str = f'Fold{num_fold}'
            model = CommModel(model_args=self.model_args, icg_approx_model=icg_approx_model).to(self.device)
            model.set_icg_approx_after_training()
            optimizer = torch.optim.Adam(model.get_icgnn_parameters(), lr=self.train_args.lr)

            with tqdm.tqdm(total=self.train_args.epochs, file=sys.stdout) as pbar:
                best_losses_n_metrics, model = \
                    self.train_and_test(dataset_type=dataset_type, data=data_split, model=model,
                                        optimizer=optimizer, epochs=self.train_args.epochs, pbar=pbar,
                                        fold_str=print_str)

            # print final
            for name in best_losses_n_metrics._fields:
                print_str += f";{name}={round(getattr(best_losses_n_metrics, name), DECIMAL)}"
            print(print_str)
            print()
            metrics_list.append(torch.tensor(best_losses_n_metrics.get_metrics()))

        metrics_matrix = torch.stack(metrics_list, dim=0)  # (F, 3)
        icgnn_final_log_and_print(metrics_matrix=metrics_matrix)
        return model

    def train_and_test(self, dataset_type: DataSet, data: Data, model: CommModel,
                       optimizer: Optimizer, epochs: int, pbar,
                       fold_str: str) -> Tuple[LossesAndMetrics, CommModel]:
        metric_type = dataset_type.get_metric_type()
        task_loss = metric_type.get_task_loss()
        best_losses_n_metrics = metric_type.get_worst_losses_n_metrics()
        best_model_state_dict = model.state_dict()
        for epoch in range(epochs):
            self.train(data=data, model=model, optimizer=optimizer, task_loss=task_loss)
            train_loss, train_metric = self.test(data=data, model=model, task_loss=task_loss, metric_type=metric_type,
                                                 mask_name='train_mask')
            val_loss, val_metric = self.test(data=data, model=model, task_loss=task_loss, metric_type=metric_type,
                                             mask_name='val_mask')
            test_loss, test_metric = self.test(data=data, model=model, task_loss=task_loss, metric_type=metric_type,
                                               mask_name='test_mask')
            losses_n_metrics = \
                LossesAndMetrics(train_loss=train_loss, val_loss=val_loss, test_loss=test_loss,
                                 train_metric=train_metric, val_metric=val_metric, test_metric=test_metric)

            # best metrics
            if metric_type.src_better_than_other(src=losses_n_metrics.val_metric,
                                                 other=best_losses_n_metrics.val_metric):
                best_losses_n_metrics = losses_n_metrics
                best_model_state_dict = model.state_dict()

            log_str = epoch_log_and_print(epoch=epoch, losses_n_metrics=losses_n_metrics,
                                          best_test_metric=best_losses_n_metrics.test_metric, fold_str=fold_str)
            pbar.set_description(log_str)
            pbar.update(n=1)

        model.load_state_dict(best_model_state_dict)
        return best_losses_n_metrics, model

    def train(self, data: Data, model: CommModel, optimizer, task_loss):
        model.train()
        optimizer.zero_grad()

        # forward
        scores = model(x=data.x.to(device=self.device), edge_index=data.edge_index.to(device=self.device))
        train_mask = data.train_mask.to(device=self.device)
        loss = task_loss(scores[train_mask], data.y.to(device=self.device)[train_mask])

        # backward
        loss.backward()
        optimizer.step()

    def test(self, data: Data, model: CommModel, task_loss, metric_type: MetricType,
             mask_name: str) -> Tuple[float, float]:
        model.eval()

        scores = model(x=data.x.to(device=self.device), edge_index=data.edge_index.to(device=self.device))
        mask = getattr(data, mask_name).to(device=self.device)

        loss = task_loss(scores[mask], data.y.to(device=self.device)[mask]).item()
        metric = metric_type.apply_metric(scores=scores[mask].detach().cpu().numpy(),
                                          target=data.y[mask.cpu()].numpy())
        return loss, metric
