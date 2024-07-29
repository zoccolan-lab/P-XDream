'''
TODO Experiment description
'''

from experiment.utils.misc import run_multi
from experiment.FeatureMapOptimization.args import ARGS
from experiment.FeatureMapOptimization.fm_optimization import FeatureMapSegmentsMultiExperiment, FeatureMapOptimizationExperiment

if __name__ == '__main__': run_multi(
    args_conf=ARGS, 
    exp_type=FeatureMapOptimizationExperiment,
    multi_exp_type=FeatureMapSegmentsMultiExperiment
)
