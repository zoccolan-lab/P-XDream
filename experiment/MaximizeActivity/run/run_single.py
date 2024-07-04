'''
TODO Experiment description
'''

import matplotlib

from experiment.MaximizeActivity.args import ARGS
from experiment.utils.misc import run_single
from experiment.MaximizeActivity.maximize_activity import MaximizeActivityExperiment

matplotlib.use('TKAgg')

if __name__ == '__main__': run_single(args_conf=ARGS, exp_type=MaximizeActivityExperiment)
