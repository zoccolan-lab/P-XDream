
from script.utils.cmdline_args import Args
from script.OptimizerTuning.args import ARGS
from script.OptimizerTuning.optimizer_tuning import OptimizationTuningExperiment, OptimizerComparisonMultiExperiment


if __name__ == '__main__':
    
    parser = Args.get_parser(args=ARGS, multirun=True)
    args   = vars(parser.parse_args())

    mrun_experiment = OptimizerComparisonMultiExperiment.from_args(
        args=args, 
        exp_type=OptimizationTuningExperiment
    )

    mrun_experiment.run()