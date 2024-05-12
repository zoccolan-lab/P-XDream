'''
This script computes the similarity between different clustering labeling
'''

import os
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score

from zdream.utils.logger import LoguruLogger
from analysis.utils.settings import FILE_NAMES, LAYER_SETTINGS, OUT_DIR, OUT_NAMES

# ------------------------------------------- SETTINGS ---------------------------------------

LAYER = 'fc8'

METRICS = {
    'RandScore'    : rand_score,
    'AdjRandScore' : adjusted_rand_score,
    'NMI'          : normalized_mutual_info_score
}

_, NAME, _, _ = LAYER_SETTINGS[LAYER]
OUT_NAME = f'{OUT_NAMES["cluster_type_comparison"]}_{NAME}'

out_dir = os.path.join(OUT_DIR, OUT_NAME)

# ---------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Initialize logger
    logger = LoguruLogger(on_file=False)
    
    # Create metric directory
    metric_dir = os.path.join(out_dir, 'metrics')
    logger.info(mess=f'Creating metrics directory to {metric_dir}')
    os.makedirs(metric_dir, exist_ok=True)

    # Create output directory for cluster similarity metrics
    logger.info(mess=f'Creating analysis target directory to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    logger.info(mess=f'')

    # Load labelings
    labelings_fp = os.path.join(out_dir, FILE_NAMES['labelings'])
    logger.info(mess=f'Loading labelings from {labelings_fp}')
    labelings = dict(np.load(labelings_fp))

    # Compute metrics and save them in distinct dataframes
    
    logger.info(mess=f'Computing {len(METRICS)}: {", ".join(METRICS.keys())}')

    indexes_df = {}

    for metric_name, metric in METRICS.items():
        
        matrix = np.ones((len(labelings), len(labelings)))  # Diagonal not set 1 as default
        labelings_ = list(labelings.values())
        
        # Iterate on the upper-triangular part of the matrix
        for i, j in itertools.combinations(range(len(labelings)), 2):
            score = metric(labelings_[i], labelings_[j])
            matrix[i, j], matrix[j, i] = score, score  # symmetric matrix

        # Create a dataframe
        index_df = pd.DataFrame(matrix, index=labelings.keys(), columns=labelings.keys())  # type: ignore
        indexes_df[metric_name] = index_df
        
    # Save the dataframes
    
    logger.info(mess='Saving dataframes.')

    for metric_name, df in indexes_df.items():
        fp = os.path.join(metric_dir, f'{metric_name}.csv')
        logger.info(mess=f' > Saving {metric_name} to {fp}')
        df.to_csv(fp)

    logger.info(mess='')
    logger.close()
    