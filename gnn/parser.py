from argparse import ArgumentParser

from additional_classes.dataset import DataSet

from gnn.classes import ModelType


def gnn_parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_type", dest="dataset_type", default=DataSet.communities,
                        type=DataSet.from_string, choices=list(DataSet), required=False)
    parser.add_argument("--model_type", dest="model_type", default=ModelType.MEAN_GNN,
                        type=ModelType.from_string, choices=list(ModelType), required=False)
    parser.add_argument('--rw_pos_length', dest='rw_pos_length', type=int, default=0, required=False)

    # model args
    parser.add_argument("--num_layers", dest="num_layers", default=3, type=int, required=False)
    parser.add_argument("--hidden_dim", dest="hidden_dim", default=8, type=int, required=False)
    parser.add_argument("--skip", dest="skip", default=False, action='store_true', required=False)
    parser.add_argument("--dropout", dest="dropout", default=0.0, type=float, required=False)
    parser.add_argument('--rw_pos_length', dest='rw_pos_length', type=int, default=0, required=False)

    # optimization
    parser.add_argument("--epochs", dest="epochs", default=300, type=int, required=False)
    parser.add_argument("--lr", dest="lr", default=1e-3, type=float, required=False)

    # reproduce
    parser.add_argument('--gpu', dest="gpu", type=int, required=False)
    return parser.parse_args()
