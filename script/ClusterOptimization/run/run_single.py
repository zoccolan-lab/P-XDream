'''
TODO Experiment description
'''

import matplotlib

from script.ClusterOptimization.cluster_optimization import ClusteringOptimizationExperiment
from script.ClusterOptimization.args import ARGS
from script.utils.cmdline_args import Args
from script.utils.misc import run_single

matplotlib.use('TKAgg')

if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=ClusteringOptimizationExperiment)
