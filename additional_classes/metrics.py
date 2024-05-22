from enum import Enum, auto
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
import torch
from typing import NamedTuple
from torchmetrics import Accuracy, AUROC, MeanAbsoluteError, F1Score
import math
from torch_geometric.data import Data
import numpy as np


class LossesAndMetrics(NamedTuple):
    train_loss: float
    val_loss: float
    test_loss: float
    train_metric: float
    val_metric: float
    test_metric: float

    def get_metrics(self):
        return [self.train_metric, self.val_metric, self.test_metric]


class MetricType(Enum):
    """
        an object for the different metrics
    """
    # classification
    ACCURACY = auto()
    AUC_ROC = auto()

    MicroF1 = auto()

    # regression
    MSE = auto()
    MAE = auto()

    def apply_metric(self, scores: np.ndarray, target: np.ndarray) -> float:
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        num_classes = None if scores.dim() == 1 else scores.size(1)  # target.max().item() + 1
        if self is MetricType.ACCURACY:
            metric = Accuracy(task="multiclass", num_classes=num_classes)
        elif self is MetricType.MSE:
            metric = MSELoss()
        elif self is MetricType.MAE:
            metric = MeanAbsoluteError()
        elif self is MetricType.AUC_ROC:
            metric = AUROC(task="multiclass", num_classes=num_classes)
        elif self is MetricType.MicroF1:
            metric = F1Score(task='multiclass', average='micro', num_classes=num_classes)
        else:
            raise ValueError(f'MetricType {self.name} not supported')

        metric = metric.to(scores.device)
        result = metric(scores, target)
        return result.item()

    def is_classification(self) -> bool:
        if self in [MetricType.AUC_ROC, MetricType.ACCURACY, MetricType.MicroF1]:
            return True
        elif self in [MetricType.MSE, MetricType.MAE]:
            return False
        else:
            raise ValueError(f'MetricType {self.name} not supported')

    def get_task_loss(self):
        if self.is_classification():
            return CrossEntropyLoss()
        elif self is MetricType.MSE:
            return MSELoss()
        elif self is MetricType.MAE:
            return L1Loss()

    def get_out_dim(self, data: Data) -> int:
        if self.is_classification():
            return int(data.y.max().item() + 1)
        else:
            return data.y.shape[-1]

    def higher_is_better(self):
        return self.is_classification()

    def src_better_than_other(self, src: float, other: float) -> bool:
        if self.higher_is_better():
            return src > other
        else:
            return src < other

    def get_worst_losses_n_metrics(self) -> LossesAndMetrics:
        if self.is_classification():
            return LossesAndMetrics(train_loss=math.inf, val_loss=math.inf, test_loss=math.inf,
                                    train_metric=-math.inf, val_metric=-math.inf, test_metric=-math.inf)
        else:
            return LossesAndMetrics(train_loss=math.inf, val_loss=math.inf, test_loss=math.inf,
                                    train_metric=math.inf, val_metric=math.inf, test_metric=math.inf)
