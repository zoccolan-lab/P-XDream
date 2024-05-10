from script.utils.cmdline_args import Args
from script.ClusteringOptimization.args import ARGS
from script.ClusteringOptimization.clustering_optimization import ClusteringOptimizationExperiment, ClusteringSubsetOptimizationMultiExperiment


if __name__ == '__main__':
    
    parser = Args.get_parser(args=ARGS, multirun=True)
    args   = vars(parser.parse_args())

    mrun_experiment = ClusteringSubsetOptimizationMultiExperiment.from_args(
        args=args, 
        exp_type=ClusteringOptimizationExperiment
    )

    mrun_experiment.run()