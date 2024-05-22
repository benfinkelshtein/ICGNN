from enum import Enum, auto
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import HeterophilousGraphDataset, WikipediaNetwork, LINKXDataset, Reddit, Reddit2
import os.path as osp
import torch
from typing import List, Optional
import copy
import numpy as np

from helpers.constants import ROOT_DIR, MAX_NUM_COMMUNITIES
from additional_classes.metrics import MetricType
from additional_classes.activation import ActivationType
from dataset_classes.communities import Communities
from dataset_classes.snap import SNAPDataset
from helpers.eigen_transform import AddLargestAbsEigenvecAndEigenval
from dataset_classes.spatio_temporal import icg_approx_load


class DataSetFamily(Enum):
    synthetic = auto()
    heterophilic = auto()
    wiki = auto()
    link = auto()
    spatio_temporal = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSetFamily[s]
        except KeyError:
            raise ValueError()


class DataSet(Enum):
    """
        an object for the different dataset_classes
    """
    # communities
    communities = auto()

    # heterophilic
    tolokers = auto()

    # wiki
    squirrel = auto()

    # link
    twitch_gamers = auto()

    # spatio temporal
    bay = auto()
    la = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()
        
    def get_family(self) -> DataSetFamily:
        if self in [DataSet.communities]:
            return DataSetFamily.synthetic
        elif self is DataSet.tolokers:
            return DataSetFamily.heterophilic
        elif self is DataSet.squirrel:
            return DataSetFamily.wiki
        elif self is DataSet.twitch_gamers:
            return DataSetFamily.link
        elif self in [DataSet.la, DataSet.bay]:
            return DataSetFamily.spatio_temporal
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')

    def is_spatio_temporal(self) -> bool:
        return self.get_family() is DataSetFamily.spatio_temporal

    def get_folds(self) -> List[int]:
        if self.get_family() in [DataSetFamily.synthetic, DataSetFamily.spatio_temporal]:
            return list(range(1))
        elif self.get_family() in [DataSetFamily.heterophilic, DataSetFamily.wiki]:
            return list(range(10))
        elif self.get_family() is DataSetFamily.link:
            return list(range(5))
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')

    def load(self, trans: bool, pos_enc_transform: Optional[T.BaseTransform] = None) -> List[Data]:
        pre_transform = None
        if not (trans and self.get_family() is DataSetFamily.spatio_temporal):
            pre_transform = T.Compose([T.ToUndirected(), AddLargestAbsEigenvecAndEigenval(k=MAX_NUM_COMMUNITIES)])
        root = osp.join(ROOT_DIR, 'datasets')
        if self is DataSet.communities:
            dataset = Communities()
            if pos_enc_transform is not None:
                dataset._data = pos_enc_transform(data=dataset._data)
                dataset._data.x = torch.cat((dataset._data.x, dataset._data.random_walk_pe), dim=1)
                delattr(dataset._data, 'random_walk_pe')
        elif self.get_family() is DataSetFamily.heterophilic:
            name = self.name.replace('_', '-').capitalize()
            dataset = HeterophilousGraphDataset(root=root, name=name, pre_transform=pre_transform)
        elif self.get_family() is DataSetFamily.wiki:
            # Same as Geom-GCN
            dataset = WikipediaNetwork(root=root, name=self.name, geom_gcn_preprocess=True,
                                       pre_transform=pre_transform)
        elif self is DataSet.twitch_gamers:
            dataset = SNAPDataset(root=root, name=self.name, pre_transform=pre_transform)
            dataset._data.y = torch.from_numpy(dataset._data.y)
        elif self.get_family() is DataSetFamily.spatio_temporal:
            dataset = icg_approx_load(dataset_name=self.name, pre_transform=pre_transform)
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')
        return dataset

    def select_fold(self, data: Data, num_fold: int) -> Data:
        if self.get_family() is DataSetFamily.synthetic:
            return data
        elif self.get_family() is DataSetFamily.wiki:
            # geom-gcn splits (60, 20, 20)
            device = data.x.device
            fold_path = osp.join(ROOT_DIR, f'folds/{self.name}_split_0.6_0.2_{num_fold}.npz')
            with np.load(fold_path) as folds_file:
                train_mask = torch.tensor(folds_file['train_mask'], dtype=torch.bool, device=device)
                val_mask = torch.tensor(folds_file['val_mask'], dtype=torch.bool, device=device)
                test_mask = torch.tensor(folds_file['test_mask'], dtype=torch.bool, device=device)

            setattr(data, 'train_mask', train_mask)
            setattr(data, 'val_mask', val_mask)
            setattr(data, 'test_mask', test_mask)

            if hasattr(data, 'non_valid_samples'):
                data.train_mask[data.non_valid_samples] = False
                data.test_mask[data.non_valid_samples] = False
                data.val_mask[data.non_valid_samples] = False
            return data
        elif self.get_family() is DataSetFamily.heterophilic:
            data_copy = copy.deepcopy(data)
            data_copy.train_mask = data_copy.train_mask[:, num_fold]
            data_copy.val_mask = data_copy.val_mask[:, num_fold]
            data_copy.test_mask = data_copy.test_mask[:, num_fold]
            return data_copy
        elif self is DataSet.twitch_gamers:
            fold_path = osp.join(ROOT_DIR, f'folds/twitch-gamer-splits.npy')
            folds_file = np.load(fold_path, allow_pickle=True)
            setattr(data, 'train_mask', torch.as_tensor(folds_file[num_fold]['train']))
            setattr(data, 'val_mask', torch.as_tensor(folds_file[num_fold]['valid']))
            setattr(data, 'test_mask', torch.as_tensor(folds_file[num_fold]['test']))
            return data
        else:
            raise ValueError(f'DataSet {self.name} not supported in select_fold')

    def get_metric_type(self) -> MetricType:
        if self.get_family() in [DataSetFamily.wiki, DataSetFamily.link]:
            return MetricType.ACCURACY
        elif self is DataSet.tolokers:
            return MetricType.AUC_ROC
        elif self.get_family() in [DataSetFamily.synthetic, DataSetFamily.spatio_temporal]:
            return MetricType.MAE
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')

    def is_communities(self) -> bool:
        return self.get_family() is DataSetFamily.synthetic

    def activation_type(self) -> ActivationType:
        if self.get_family() is DataSetFamily.heterophilic:
            return ActivationType.GELU
        else:
            return ActivationType.RELU
