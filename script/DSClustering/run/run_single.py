
from script.DSClustering.parser import get_parser
from script.DSClustering.ds_clustering import DSClusteringExperiment

def main(args):

    experiment = DSClusteringExperiment.from_args(args=args)
    experiment.run()

if __name__ == '__main__':
    
    parser = get_parser()
    
    args = vars(parser.parse_args())
    
    main(args=args)
