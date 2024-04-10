"""
TODO Experiment description
"""

import matplotlib

from script.ClusteringOptimization.clustering_optimization import ClusteringOptimizationExperiment
from script.ClusteringOptimization.parser import get_parser
from script.script_utils import run_single

matplotlib.use('TKAgg')

if __name__ == '__main__':

    parser = get_parser()
    
    args = vars(parser.parse_args())
    
    run_single(
        args=args,
        exp_type=ClusteringOptimizationExperiment
    )
