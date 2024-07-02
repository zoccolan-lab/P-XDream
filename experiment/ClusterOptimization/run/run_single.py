'''
TODO Experiment description
'''

import matplotlib

from experiment.ClusterOptimization.cluster_optimization import ClusteringOptimizationExperiment
from experiment.ClusterOptimization.args import ARGS
from experiment.utils.cmdline_args import Args
from experiment.utils.misc import run_single

matplotlib.use('TKAgg')

if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=ClusteringOptimizationExperiment)
