from script.utils.cmdline_args import Args
from script.ClusterOptimization.args import ARGS
from script.ClusterOptimization.cluster_optimization import ClusteringOptimizationExperiment, ClustersBestStimuliMultiExperiment


if __name__ == '__main__':
    
    parser = Args.get_parser(args=ARGS, multirun=True)
    args   = vars(parser.parse_args())

    mrun_experiment = ClustersBestStimuliMultiExperiment.from_args(
        args=args, 
        exp_type=ClusteringOptimizationExperiment
    )

    mrun_experiment.run()