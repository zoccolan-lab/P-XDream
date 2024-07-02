from experiment.utils.cmdline_args import Args
from experiment.ClusterOptimization.args import ARGS
from experiment.ClusterOptimization.cluster_optimization import ClusteringOptimizationExperiment, ClustersBestStimuliMultiExperiment


if __name__ == '__main__':
    
    parser = Args.get_parser(args=ARGS, multirun=True)
    args   = vars(parser.parse_args())

    mrun_experiment = ClustersBestStimuliMultiExperiment.from_args(
        args=args, 
        exp_type=ClusteringOptimizationExperiment
    )

    mrun_experiment.run()