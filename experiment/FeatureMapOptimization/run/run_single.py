'''
TODO Experiment description
'''

from experiment.FeatureMapOptimization.args import ARGS
from experiment.FeatureMapOptimization.fm_optimization import FeatureMapOptimizationExperiment
from experiment.utils.misc import run_single

if __name__ == '__main__': run_single(args_conf=ARGS, exp_type=FeatureMapOptimizationExperiment)
