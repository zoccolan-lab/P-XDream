import os

import numpy as np
from   numpy.typing import NDArray
import pandas as pd
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score

from zdream.utils.logger import LoguruLogger
from zdream.utils.io_ import read_json

# ------------------------------------------- SETTINGS ---------------------------------------

SETTINGS_FILE = os.path.abspath(os.path.join(__file__, '..', '..', 'local_settings.json'))
settings = read_json(SETTINGS_FILE)

LAYER = 'fc7-relu'

SETTINGS = {
    'fc8'     : 'alexnetfc8',
    'fc7-relu': 'alexnetfc7relu',
    'fc7'     : 'alexnetfc7'
}

NAME = SETTINGS[LAYER]

OUT_DIR     = settings['out_dir']

FILE_NAMES = {
    'labelings' : 'labelings.npz'
}

OUT_NAME = f'clusterings_labelings_{NAME}'  # the directory is supposed to contain the labelings

K        =        3 # Number of points to skip for the text
FIGSIZE  = (10, 10)
FONTSIZE =        6

METRICS = {
    'RandScore'    : rand_score,
    'AdjRandScore' : adjusted_rand_score,
    'NMI'          : normalized_mutual_info_score
}

# ---------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # 0. Creating logger and output directory

    logger = LoguruLogger(on_file=False)

    out_dir = os.path.join(OUT_DIR, OUT_NAME)
    logger.info(mess=f'Creating analysis target directory to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    logger.info(mess=f'')

    # 1. LOADING LABELINGS

    labelings_fp = os.path.join(out_dir, FILE_NAMES['labelings'])
    logger.info(mess=f'Loading labelings from {labelings_fp}')
    labelings = dict(np.load(labelings_fp, allow_pickle=True))

    # 2. COMPUTING METRICS
    
    logger.info(mess=f'Computing {len(METRICS)}: {", ".join(METRICS.keys())}')

    indexes_df = {}

    for metric_name, metric in METRICS.items():
        
        matrix = np.ones((len(labelings), len(labelings)))
        
        labelings_ = list(labelings.values())
        for i in range(len(labelings)):
            for j in range(i+1, len(labelings)):
                labeling1 = labelings_[i]
                labeling2 = labelings_[j]
                matrix[i, j] = metric(labeling1, labeling2)

        index_df = pd.DataFrame(matrix, index=labelings.keys(), columns=labelings.keys())  # type: ignore
        indexes_df[metric_name] = index_df
        
    # 3. Saving
    
    logger.info(mess='Saving dataframes.')

    metric_dir = os.path.join(out_dir, 'metrics')
    logger.info(mess=f'Creating metrics directory to {metric_dir}')
    os.makedirs(metric_dir, exist_ok=True)

    # Save the dataframes
    for metric_name, df in indexes_df.items():
        fp = os.path.join(metric_dir, f'{metric_name}.csv')
        logger.info(mess=f' > Saving {metric_name} to {fp}')
        df.to_csv(fp)

    logger.info(mess='')
    logger.close()