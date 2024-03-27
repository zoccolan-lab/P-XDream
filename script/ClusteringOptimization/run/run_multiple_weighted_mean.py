from script.ClusteringOptimization.clustering_optimization import ClusteringOptimizationExperiment, UnitsWeightingMultiExperiment
from script.ClusteringOptimization.parser import get_parser
from script.multiexperiment_parsing import parse_multiexperiment_args
from zdream.utils.io_ import read_json
from zdream.utils.misc import flatten_dict


def main(args):
    
    json_conf, args_conf = parse_multiexperiment_args(args=args)
    
    mrun_experiment = UnitsWeightingMultiExperiment(
        experiment=ClusteringOptimizationExperiment,
        default_conf=json_conf,
        experiment_conf=args_conf
    )

    mrun_experiment.run()

if __name__ == '__main__':
    
    parser = get_parser(multirun=True)
    
    conf = vars(parser.parse_args())

    main(conf)