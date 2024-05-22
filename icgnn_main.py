from icgnn.parser import icgnn_parse_arguments
from icgnn.experiments import TransExperiment

if __name__ == '__main__':
    args = icgnn_parse_arguments()
    TransExperiment(args=args).run()
