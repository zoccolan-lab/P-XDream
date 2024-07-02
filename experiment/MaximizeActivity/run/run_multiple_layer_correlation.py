
from experiment.utils.cmdline_args import Args
from experiment.MaximizeActivity.args import ARGS
from experiment.MaximizeActivity.maximize_activity import MaximizeActivityExperiment2, LayersCorrelationMultiExperiment


if __name__ == '__main__':
    
    parser = Args.get_parser(args=ARGS, multirun=True)
    args   = vars(parser.parse_args())

    mrun_experiment = LayersCorrelationMultiExperiment.from_args(
        args=args, 
        exp_type=MaximizeActivityExperiment2
    )

    mrun_experiment.run()