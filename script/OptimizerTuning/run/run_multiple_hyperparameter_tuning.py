from script.OptimizerTuning.parser import get_parser
from script.OptimizerTuning.optimizer_tuning import OptimizationTuningExperiment, HyperparameterTuningMultiExperiment
from zdream.utils.io_ import read_json
from zdream.utils.misc import flatten_dict

if __name__ == '__main__':
    
    parser = get_parser(multirun=True)
    
    parser.add_argument('--hyperparameter', type=str)
    
    args = vars(parser.parse_args())
    
    hyperparameter = args.pop('hyperparameter')

    mrun_experiment = HyperparameterTuningMultiExperiment.from_args(
        args=args,
        exp_type=OptimizationTuningExperiment
    )
    
    mrun_experiment.hyperparameter = hyperparameter

    mrun_experiment.run()