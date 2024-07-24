from experiment.ClusteringAlgo.args import ARGS
from experiment.ClusteringAlgo.clustering_algo import DBSCANClusteringExperiment
from experiment.ClusteringAlgo.args import ARGS
from experiment.utils.misc import run_single


if __name__ == '__main__':  run_single(args_conf=ARGS, exp_type=DBSCANClusteringExperiment)