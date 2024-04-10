from script.ClusteringOptimization.clustering_optimization import ClusteringOptimizationExperiment, ClusteringSubsetOptimizationMultiExperiment
from script.ClusteringOptimization.parser import get_parser
from zdream.utils.io_ import read_json
from zdream.utils.misc import flatten_dict


def main(args):
        
    mrun_experiment = ClusteringSubsetOptimizationMultiExperiment.from_args(
        args=args,
        exp_type=ClusteringOptimizationExperiment
    )

    mrun_experiment.run()

if __name__ == '__main__':
    
    parser = get_parser(multirun=True)
    conf = vars(parser.parse_args())

    main(conf)