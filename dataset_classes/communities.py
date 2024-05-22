import torch
import numpy as np

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.nn.conv.simple_conv import SimpleConv

# topology global vars
COMMUNITY_SIZE = 200
NUM_COMMUNITIES = 5
SMALL_GRAPH_SIZE = 30

# feature global vars
NUM_MP = 3
ADJ_NOISE_RATIO = 0.05
FEAT_NOISE_RATIO = 0.5


class Communities(InMemoryDataset):
    def __init__(self):
        super().__init__()
        self.num_communities = NUM_COMMUNITIES
        self.num_mp = NUM_MP
        self.small_graph_size = SMALL_GRAPH_SIZE
        self.adj_noise_ratio = ADJ_NOISE_RATIO
        self.feat_noise_ratio = FEAT_NOISE_RATIO

        self._data = self.create_data()

    def gett(self, idx: int) -> Data:
        return self._data

    def create_small_graph(self):
        com_scale = torch.arange(start=1, end=self.num_communities + 1) * 0.1 - 0.2

        # start and end position of each community
        community_start = np.random.choice(self.small_graph_size, size=(self.num_communities,), replace=True)
        community_start = torch.from_numpy(community_start)
        community_end = np.random.choice(self.small_graph_size - 1, size=(self.num_communities,), replace=True)
        community_end = torch.from_numpy(community_end)
        community_end[community_end == community_start] = self.small_graph_size - 1

        # create adj mat
        affiliate_vec = torch.zeros(size=(self.small_graph_size, 1))
        adj_mat = torch.zeros(size=(self.small_graph_size, self.small_graph_size))
        affiliate_mat = []
        for community_idx in range(self.num_communities):
            start = community_start[community_idx]
            end = community_end[community_idx]

            # edge probabilities
            if start < end:
                affiliate_vec[start: end, 0] = 1.0
            else:
                affiliate_vec[start:, 0] = 1.0
                affiliate_vec[:end, 0] = 1.0
            adj_mat += (affiliate_vec * com_scale[community_idx]) @ affiliate_vec.T
            affiliate_mat.append(affiliate_vec.squeeze(dim=1))
            affiliate_vec = affiliate_vec * 0
        norm_com_scale = (com_scale - adj_mat.min()) / (adj_mat.max() - adj_mat.min())
        adj_mat = (adj_mat - adj_mat.min()) / (adj_mat.max() - adj_mat.min())
        affiliate_mat = torch.stack(affiliate_mat, dim=1)

        # map graph nodes to small graph nodes
        nodes_to_small_nodes = torch.empty(size=(0,))
        for start, end in zip(community_start, community_end):
            # community node to base nodes dictionary
            if start < end:
                small_graph_comm = np.arange(start=start, stop=end)
            else:
                small_graph_comm = np.concatenate((np.arange(start=0, stop=end),
                                                   np.arange(start=start, stop=self.small_graph_size)), axis=0)
            nodes_to_small_nodes_per_comm = torch.from_numpy(np.random.choice(small_graph_comm,
                                                                              size=(COMMUNITY_SIZE,)))
            nodes_to_small_nodes = torch.concatenate((nodes_to_small_nodes, nodes_to_small_nodes_per_comm), dim=0)
        nodes_to_small_nodes = nodes_to_small_nodes.long()  # (N,)
        return nodes_to_small_nodes, adj_mat, norm_com_scale, affiliate_mat

    def create_data(self) -> Data:
        nodes_to_small_nodes, small_adj_mat, norm_com_scale, small_affiliate_mat = self.create_small_graph()
        adj_mat = small_adj_mat[nodes_to_small_nodes][:, nodes_to_small_nodes]

        # add NOISE to adj
        adj_mat += self.adj_noise_ratio * torch.randn(size=adj_mat.size())
        adj_mat[adj_mat > 1] = 0
        adj_mat[adj_mat < 0] = 0

        # sample adj
        adj_mat = torch.triu(torch.bernoulli(adj_mat))
        adj_mat = adj_mat + adj_mat.T - torch.diag(torch.diag(adj_mat))
        edge_index = dense_to_sparse(adj=adj_mat)[0]

        # regression target - the community scale of each node
        affiliate_mat = small_affiliate_mat[nodes_to_small_nodes]
        y = 10 * affiliate_mat * norm_com_scale  # (N, K)

        # add NOISE to features
        x = y.clone()
        std_x = torch.std(x, dim=0)  # (K,)
        x = x + self.feat_noise_ratio * std_x.unsqueeze(dim=0) * torch.randn(size=x.size())  # (N, K)

        # propagate features
        simple_conv = SimpleConv(aggr='mean')
        for _ in range(self.num_mp):
            x = simple_conv(x=x, edge_index=edge_index)

        # create masks for train/val/test
        num_nodes = x.shape[0]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[::3] = 1
        val_mask[1::3] = 1
        test_mask[2::3] = 1

        return Data(x=x, edge_index=edge_index.long(), y=y,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


if __name__ == '__main__':
    dataset = Communities()
    print()
