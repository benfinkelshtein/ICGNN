from torch_geometric.data import Data
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay
from torch.utils.data import DataLoader
import os.path as osp
from torch_geometric.transforms import BaseTransform
from typing import List, Tuple, Optional

from helpers.constants import ROOT_DIR


def load_dataset_by_name(dataset_name: str):
    dataset_path = osp.join(ROOT_DIR, 'datasets')
    if dataset_name == 'la':
        dataset = MetrLA(root=dataset_path)
    elif dataset_name == 'bay':
        dataset = PemsBay(root=dataset_path, mask_zeros=True)
    else:
        raise ValueError(f'DataSet {dataset_name} not supported in icg_approx_load')
    return dataset


def icg_approx_load(dataset_name: str, pre_transform: Optional[BaseTransform] = None, batch_size: int = 32, window: int = 24,
                horizon: int = 3, stride: int = 1, threshold: int = 0.1) -> List[Data]:
    dataset = load_dataset_by_name(dataset_name=dataset_name)
    splitter = dataset.get_splitter(val_len=0.1, test_len=0.2)
    connectivity = dataset.get_connectivity(threshold=threshold,
                                            force_symmetric=True,
                                            include_self=False)  # True if you want self loops
    tsl_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                        connectivity=connectivity,
                                        mask=dataset.mask,
                                        # transform=ToPyG(),
                                        window=window,
                                        horizon=horizon,
                                        stride=stride)
    transform = {
        'target': StandardScaler(axis=(0, 1))
    }
    dm = SpatioTemporalDataModule(
        dataset=tsl_dataset,
        scalers=transform,
        splitter=splitter,
        batch_size=batch_size,
        workers=0
    )
    dm.setup()

    data = dm.trainset.dataset[dm.trainset.indices]
    if pre_transform is not None:
        data = pre_transform.forward(data=data)
    return [data]


def icgnn_load(dataset_name: str, batch_size: int = 32, window: int = 24, horizon: int = 3,
               stride: int = 1, threshold: int = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = load_dataset_by_name(dataset_name=dataset_name)
    splitter = dataset.get_splitter(val_len=0.1, test_len=0.2)
    connectivity = dataset.get_connectivity(threshold=threshold,
                                            include_self=True)  # True if you want self loops
    tsl_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                        connectivity=connectivity,
                                        mask=dataset.mask,
                                        window=window,
                                        horizon=horizon,
                                        stride=stride)
    transform = {
        'target': StandardScaler(axis=(0, 1))
    }
    dm = SpatioTemporalDataModule(
        dataset=tsl_dataset,
        scalers=transform,
        splitter=splitter,
        batch_size=batch_size,
        workers=0
    )
    dm.setup()

    return dm.get_dataloader(split='train'), dm.get_dataloader(split='val'), dm.get_dataloader(split='test')
