from icg_approximation.parser import icg_approx_parse_arguments
from icg_approximation.experiments import DecompExperiment

if __name__ == '__main__':
    args = icg_approx_parse_arguments()
    DecompExperiment(args=args).run()
