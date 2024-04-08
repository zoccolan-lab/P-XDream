"""
TODO Experiment description
"""

import matplotlib

from script.OptimizerTuning.parser import get_parser
from script.OptimizerTuning.optimizer_tuning import OptimizationTuningExperiment
from script.script_utils import run_single

matplotlib.use('TKAgg')

if __name__ == '__main__':

    parser = get_parser()
    args = vars(parser.parse_args())
    
    run_single(
        args=args,
        exp_type=OptimizationTuningExperiment
    )
