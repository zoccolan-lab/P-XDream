'''
TODO Experiment description
'''

import matplotlib

from script.cmdline_args import Args
from script.script_utils import run_single
from script.MaximizeActivity.args import ARGS
from script.MaximizeActivity.maximize_activity import MaximizeActivityRFMapsExperiment

matplotlib.use('TKAgg')

if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=MaximizeActivityRFMapsExperiment)
