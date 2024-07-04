'''
TODO Experiment description
'''

import matplotlib

from experiment.ClusterOptimization.cluster_optimization import ClusteringOptimizationExperiment
from experiment.ClusterOptimization.args import ARGS
from experiment.utils.misc import run_single
from zdream.utils.parameters import ArgParams

matplotlib.use('TKAgg')

if __name__ == '__main__':
    
    run_single(args_conf=args, exp_type=ClusteringOptimizationExperiment)
