'''
TODO Experiment description
'''

from experiment.utils.misc import run_multi
from experiment.ClusterOptimization.args import ARGS
from experiment.ClusterOptimization.cluster_optimization import ClusteringOptimizationExperiment, ClusterSubsettingOptimizationMultiExperiment

if __name__ == '__main__': run_multi(
    args_conf=ARGS, 
    exp_type=ClusteringOptimizationExperiment,
    multi_exp_type=ClusterSubsettingOptimizationMultiExperiment
)
