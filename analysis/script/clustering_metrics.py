''
import itertools
import os

import numpy as np
import pandas as pd

from analysis.utils.misc import load_clusters
from analysis.utils.settings import ALEXNET_DIR, LAYER_SETTINGS, OUT_DIR
from experiment.utils.misc import make_dir
from zdream.clustering.cluster import Clusters
from zdream.utils.logger import LoguruLogger

# --- SETTINGS ---

LAYER = 'conv5-maxpool'

out_dir = os.path.join(OUT_DIR, "clustering_analysis", "metrics", LAYER_SETTINGS[LAYER]['directory'])
clu_dir = os.path.join(ALEXNET_DIR, LAYER_SETTINGS[LAYER]['directory'])

METRICS = {
    'RandScore'    : Clusters.clusters_rand_score,
    'AdjRandScore' : Clusters.clusters_adjusted_rand_score,
    'NMI'          : Clusters.clusters_normalized_mutual_info_score
}

def main():

    # Initialize logger
    logger = LoguruLogger(on_file=False)
    
    # Load clusters
    clusters = load_clusters(dir=clu_dir, logger=logger)
        
    # Create metric directory
    metric_dir = make_dir(out_dir)
    
    logger.info(mess=f'Computing {len(METRICS)}: {", ".join(METRICS.keys())}')

    indexes_df = {}

    for metric_name, metric in METRICS.items():
        
        cluster_names = list(clusters.keys())
        
        matrix = np.ones((len(clusters), len(clusters)))  # Diagonal not set 1 as default
        
        # Iterate on the upper-triangular part of the matrix
        for i, j in itertools.combinations(range(len(clusters)), 2):
            score = metric(clusters[cluster_names[i]], clusters[cluster_names[j]])
            matrix[i, j], matrix[j, i] = score, score  # symmetric matrix

        # Create a dataframe
        index_df = pd.DataFrame(matrix, index=cluster_names, columns=cluster_names)
        indexes_df[metric_name] = index_df
        
    # Save the dataframes
    logger.info(mess='Saving dataframes.')

    for metric_name, df in indexes_df.items():
        fp = os.path.join(metric_dir, f'{metric_name}.csv')
        logger.info(mess=f' > Saving {metric_name} to {fp}')
        df.to_csv(fp)

    logger.info(mess='')
    logger.close()
    
if __name__ == '__main__': 
    
    for layer in LAYER_SETTINGS.keys():
        
        LAYER = layer
        LAYER = 'conv5-maxpool'

        out_dir = os.path.join(OUT_DIR, "clustering_analysis", "metrics", LAYER_SETTINGS[LAYER]['directory'])
        clu_dir = os.path.join(ALEXNET_DIR, LAYER_SETTINGS[LAYER]['directory'])    
        
        main()
    
    #main()