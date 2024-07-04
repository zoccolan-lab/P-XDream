from experiment.ClusteringAlgo.args import ARGS
from experiment.ClusteringAlgo.clustering_algo import NCClusteringExperiment
from experiment.utils.args import Args
from experiment.utils.misc import run_single


if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=NCClusteringExperiment)

