'''
TODO Experiment description
'''

import matplotlib

from experiment.MaximizeActivity.args import ARGS
from experiment.utils.misc import run_multi
from experiment.MaximizeActivity.maximize_activity import MaximizeActivityExperiment, NeuronScalingMultiExperiment

#matplotlib.use('TKAgg')

if __name__ == '__main__': run_multi(
    args_conf=ARGS, 
    exp_type=MaximizeActivityExperiment,
    multi_exp_type=NeuronScalingMultiExperiment
)