from script.OptimizerTuning.parser import get_parser
from script.OptimizerTuning.optimizer_tuning import OptimizationTuningExperiment, OptimizerComparisonMultiExperiment
from zdream.utils.io_ import read_json
from zdream.utils.misc import flatten_dict

if __name__ == '__main__':
    
    parser = get_parser(multirun=True)
    
    args = vars(parser.parse_args())

    mrun_experiment = OptimizerComparisonMultiExperiment.from_args(
        args=args,
        exp_type=OptimizationTuningExperiment
    )

    mrun_experiment.run()