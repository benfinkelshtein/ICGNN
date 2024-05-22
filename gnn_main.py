from gnn.parser import gnn_parse_arguments
from gnn.experiments import GNNExperiment

if __name__ == '__main__':
    args = gnn_parse_arguments()
    GNNExperiment(args=args).run()
