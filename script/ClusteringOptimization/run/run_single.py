'''
TODO Experiment description
'''

import matplotlib

from script.ClusteringOptimization.args import ARGS
from script.ClusteringOptimization.clustering_optimization import ClusteringOptimizationExperiment
from script.utils.cmdline_args import Args
from script.utils.utils import run_single

matplotlib.use('TKAgg')

if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=ClusteringOptimizationExperiment)
