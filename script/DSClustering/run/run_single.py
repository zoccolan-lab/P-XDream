from script.cmdline_args import Args
from script.script_utils import run_single
from script.DSClustering.args import ARGS
from script.DSClustering.ds_clustering import DSClusteringExperiment


if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=DSClusteringExperiment)

