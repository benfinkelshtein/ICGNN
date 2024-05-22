from argparse import ArgumentParser

from additional_classes.dataset import DataSet


def icg_approx_parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_type", dest="dataset_type", default=DataSet.communities,
                        type=DataSet.from_string, choices=list(DataSet), required=False)

    # icg_approximation train args
    parser.add_argument("--icg_approx_epochs", dest="icg_approx_epochs", default=1000, type=int, required=False)
    parser.add_argument("--icg_approx_lr", dest="icg_approx_lr", default=1e-1, type=float, required=False)
    parser.add_argument("--loss_scale", dest="loss_scale", default=0.0, type=float, required=False)
    parser.add_argument("--cut_norm", dest="cut_norm", default=False, action='store_true', required=False)
    parser.add_argument("--add_eigen", dest="add_eigen", default=False, action='store_true', required=False)
    parser.add_argument("--node_drop_ratio", dest="node_drop_ratio", default=0.0, type=float, required=False)

    # icg_approximation args
    parser.add_argument("--num_communities", dest="num_communities", default=5, type=int, required=False)
    parser.add_argument("--encode_dim", dest="encode_dim", default=0, type=int, required=False)

    # reproduce
    return parser.parse_args()
