import torch
from torch import Tensor

from helpers.constants import DECIMAL
from additional_classes.metrics import LossesAndMetrics


def epoch_log_and_print(epoch: int, losses_n_metrics: LossesAndMetrics, best_test_metric, fold_str: str) -> str:
    log_str = fold_str + f';MLP;epoch:{epoch}'
    for name in losses_n_metrics._fields:
        log_str += f";{name}={round(getattr(losses_n_metrics, name), DECIMAL)}"
    log_str += f"({round(best_test_metric, DECIMAL)})"
    return log_str


def icgnn_final_log_and_print(metrics_matrix: Tensor):
    metrics_mean = torch.mean(metrics_matrix, dim=0).tolist()  # (3,)
    num_folds = metrics_matrix.shape[0]
    if num_folds > 1:
        metrics_std = torch.std(metrics_matrix, dim=0).tolist()  # (3,)

    print_str = "Final "
    for idx, split in enumerate(['train', 'val', 'test']):
        print_str += f'{split}={round(metrics_mean[idx], DECIMAL)}'
        if num_folds > 1:
            print_str += f'+-{round(metrics_std[idx], DECIMAL)}'
        print_str += ';'
    print(print_str[:-1])
