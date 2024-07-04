from experiment.utils.args import Args
from experiment.utils.misc import run_single
from experiment.OptimizerTuning.args import ARGS
from experiment.OptimizerTuning.optimizer_tuning import OptimizationTuningExperiment


if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=OptimizationTuningExperiment)

