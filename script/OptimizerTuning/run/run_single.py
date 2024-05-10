from script.utils.cmdline_args import Args
from script.utils.utils import run_single
from script.OptimizerTuning.args import ARGS
from script.OptimizerTuning.optimizer_tuning import OptimizationTuningExperiment


if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=OptimizationTuningExperiment)

