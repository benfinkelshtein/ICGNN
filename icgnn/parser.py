from argparse import ArgumentParser

from additional_classes.dataset import DataSet
from icgnn.classes import TransType


def icgnn_parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_type", dest="dataset_type", default=DataSet.communities,
                        type=DataSet.from_string, choices=list(DataSet), required=False)

    # icg_approximation train args
    parser.add_argument("--icg_approx_epochs", dest="icg_approx_epochs", default=1000, type=int, required=False)
    parser.add_argument("--icg_approx_lr", dest="icg_approx_lr", default=1e-1, type=float, required=False)
    parser.add_argument("--loss_scale", dest="loss_scale", default=0.0, type=float, required=False)
    parser.add_argument("--add_eigen", dest="add_eigen", default=False, action='store_true',
                        required=False)
    parser.add_argument("--node_drop_ratio", dest="node_drop_ratio", default=0.0, type=float, required=False)

    # icg_approximation args
    parser.add_argument("--num_communities", dest="num_communities", default=5, type=int, required=False)
    parser.add_argument("--encode_dim", dest="encode_dim", default=0, type=int, required=False)

    # icgnn args
    parser.add_argument("--num_layers", dest="num_layers", default=3, type=int, required=False)
    parser.add_argument("--icgnn_type", dest="icgnn_type", default=TransType.Matrix,
                        type=TransType.from_string, choices=list(TransType), required=False)
    parser.add_argument("--nn_num_layers", dest="nn_num_layers", default=1, type=int, required=False)
    parser.add_argument("--hidden_dim", dest="hidden_dim", default=8, type=int, required=False)
    parser.add_argument("--dropout", dest="dropout", default=0.0, type=float, required=False)
    parser.add_argument("--skip", dest="skip", default=False, action='store_true', required=False)

    # icgnn train args
    parser.add_argument("--epochs", dest="epochs", default=100, type=int, required=False)
    parser.add_argument("--lr", dest="lr", default=1e-3, type=float, required=False)
    return parser.parse_args()
