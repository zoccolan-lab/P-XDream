from script.ClusteringAlgo.args import ARGS
from script.ClusteringAlgo.clustering_algo import NCClusteringExperiment
from script.utils.cmdline_args import Args
from script.utils.misc import run_single


if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=NCClusteringExperiment)

