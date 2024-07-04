
from experiment.utils.args import Args
from experiment.OptimizerTuning.args import ARGS
from experiment.OptimizerTuning.optimizer_tuning import OptimizationTuningExperiment, OptimizerComparisonMultiExperiment


if __name__ == '__main__':
    
    parser = Args.get_parser(args=ARGS, multirun=True)
    parser.add_argument('--hyperparameter', type=str, default=1)
    args   = vars(parser.parse_args())
    
    hyperparameter = args['hyperparameter']
    args.pop('hyperparameter')

    mrun_experiment = OptimizerComparisonMultiExperiment.from_args(
        args=args, 
        exp_type=OptimizationTuningExperiment
    )
    
    setattr(mrun_experiment, 'hyperparameter', hyperparameter)

    mrun_experiment.run()