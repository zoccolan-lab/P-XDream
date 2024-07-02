'''
TODO Experiment description
'''

import matplotlib

from experiment.utils.cmdline_args import Args
from experiment.utils.misc import run_single
from experiment.MaximizeActivity.args import ARGS
from experiment.MaximizeActivity.maximize_activity import MaximizeActivityRFMapsExperiment

matplotlib.use('TKAgg')

if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=MaximizeActivityRFMapsExperiment)
