import torch
import pandas as pd
from os import path

from torch_geometric.data import Data

# taken from https://github.com/CUAI/Non-Homophily-Large-Scale/tree/master


def load_twitch_gamer_dataset(raw_dir: str, task="mature", normalize=True):
    edges = pd.read_csv(path.join(raw_dir, 'large_twitch_edges.csv'))
    nodes = pd.read_csv(path.join(raw_dir, 'large_twitch_features.csv'))
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = Data(x=node_feat, edge_index=edge_index, num_nodes=num_nodes, y=label)
    return [dataset]


def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding

    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()

    return label, features

