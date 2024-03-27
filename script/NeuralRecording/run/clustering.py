from argparse import ArgumentParser
from zdream.clustering.model import AffinityMatrix
from zdream.clustering.algo import BaseDSClustering
from zdream.logger import LoguruLogger

def main(aff_mat_fp: str, out_dir: str):
    
    # Load affinity matrix
    aff_mat = AffinityMatrix.from_file(path=aff_mat_fp)
    
    # Logger
    logger = LoguruLogger(on_file=False)
    
    # Algorithm
    algo = BaseDSClustering(
        aff_mat=aff_mat, 
        max_iter=50000, 
        logger=logger
    )
    algo.run()
    
    # Save
    algo.clusters.dump(
        out_fp=out_dir,
        logger=logger
    )
    
if __name__ == '__main__':
    
    parser = ArgumentParser()
    
    # Configuration file
    parser.add_argument('--affinity_matrix', type=str,   help='Path to the affinity matrix for clustering')
    parser.add_argument('--out_dir',         type=str,   help='Path to save clustering results')
    
    conf = vars(parser.parse_args())
    
    main(
        aff_mat_fp = conf['affinity_matrix'],
        out_dir    = conf['out_dir']
    )
    
    