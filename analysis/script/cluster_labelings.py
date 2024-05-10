import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering

from zdream.utils.io_ import read_json
from zdream.utils.logger import LoguruLogger
from zdream.clustering.ds import DSClusters

# ------------------------------------------- SETTINGS ---------------------------------------

LAYER = 'fc8'

SETTINGS = {
    'fc8'     : ('fc8',      'alexnetfc8',     True,    50),
    'fc7-relu': ('fc7-relu', 'alexnetfc7relu', False,  125),
    'fc7'     : ('fc7',      'alexnetfc7',     False,   56)
}


CLU_DIR, NAME, TRUE_CLASSES, N_CLU = SETTINGS[LAYER]

SETTINGS_FILE = os.path.abspath(os.path.join(__file__, '..', '..', 'local_settings.json'))
settings = read_json(SETTINGS_FILE)

OUT_DIR     = settings['out_dir']
WORDNET_DIR = settings['wordnet_dir']
CLUSTER_DIR = os.path.join(settings['cluster_dir'], CLU_DIR)

FILE_NAMES = {
    'recordings'      : 'recordings.npy',
    'affinity_matrix' : 'affinity_matrix.npy',
    'ds_clusters'     : 'DSClusters.json',
    'true_classes'    : 'imagenet_superclass_index.json'
}

OUT_NAME = f'clusterings_labelings_{NAME}'

PCA_DIM      = 500   # Number of PCA components for Gaussian Mixture 

# ---------------------------------------------------------------------------------------------

def start(logger, name):
    logger.info(mess=name)
    logger.formatting = lambda x: f'> {x}'

def done(logger):
    logger.info(mess='Done')
    logger.reset_formatting()
    logger.info(mess='')

if __name__ == '__main__':

    # 0. Creating logger and output directory

    logger = LoguruLogger(on_file=False)

    out_dir = os.path.join(OUT_DIR, OUT_NAME)
    logger.info(mess=f'Creating analysis target directory to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    logger.info(mess=f'')

    # 1. DOMINANT SET

    start(logger, 'DOMINANT SET Clustering')

    ds_file = os.path.join(CLUSTER_DIR, FILE_NAMES['ds_clusters'])
    ds      = DSClusters.from_file(ds_file, logger=logger)

    logger.info(mess='Retrieving labeling')
    ds_labeling = ds.labeling

    done(logger)

    # 2. GAUSSIAN MIXTURE

    start(logger, 'GAUSSIAN MIXTURE Clustering')

    recordings_fp = os.path.join(CLUSTER_DIR, FILE_NAMES['recordings'])
    logger.info(mess=f'Loading recordings from {recordings_fp}')
    recordings = np.load(recordings_fp)

    logger.info(mess=f'Applying PCA from {recordings.shape[-1]} to {PCA_DIM}')
    pca = PCA(n_components=PCA_DIM)
    recordings_pca = pca.fit_transform(recordings)

    logger.info(mess=f'Fitting Gaussian Mixture with {N_CLU} clusters')
    gmm = GaussianMixture(n_components=N_CLU)
    gmm_labeling = gmm.fit_predict(recordings_pca)

    done(logger)

    # 3. NORMALIZED CUT

    start(logger, 'NORMALIZED CUT Clustering')

    affinity_matrix_fp = os.path.join(CLUSTER_DIR, FILE_NAMES['affinity_matrix'])
    logger.info(mess=f'Loading affinity matrix from {affinity_matrix_fp}')
    aff_mat = np.load(affinity_matrix_fp)

    logger.info(mess=f'Fitting Spectral Clustering with {N_CLU} clusters')
    nc = SpectralClustering(n_clusters=N_CLU, affinity='precomputed')
    nc_labeling = nc.fit_predict(aff_mat)

    done(logger)

    # 4. ADJACENT CLUSTERING

    start(logger, 'ADJACENT Clustering')

    n_obj = aff_mat.shape[0]

    # Create a trivial clustering assigning same labeling to contiguous objects
    trivial_labeling = np.zeros(n_obj)
    clu_size = n_obj // N_CLU

    logger.info(mess=f'Creating trivial clustering with {N_CLU} clusters')
    for i in range(N_CLU):
        if (i+1)*clu_size > n_obj: trivial_labeling[i*clu_size:] = i  # Prevent out of bounds
        else:                      trivial_labeling[i*clu_size:(i+1)*clu_size] = i
        
    trivial_labeling = [int(n) for n in trivial_labeling]
        
    done(logger)
    
    # 5. RANDOM

    start(logger, 'RANDOM Clustering')

    n_obj = aff_mat.shape[0]

    # Create a trivial clustering assigning same labeling to contiguous objects
    
    random_labeling = [int(n) for n in np.random.permutation(trivial_labeling)]

    done(logger)

    # 6. TRUE CLASSES

    if TRUE_CLASSES:

        start(logger, 'TRUE CLASSES')

        true_classes_fp = os.path.join(WORDNET_DIR, FILE_NAMES['true_classes'])
        logger.info(mess=f'Loading true labels from {true_classes_fp}')
        true_classes = read_json(true_classes_fp)

        # Extract unique labels and assign a cluster index to each label
        unique_classes = list(set([a for _, a in true_classes.values()]))
        logger.info(mess=f'Extracting label of {len(unique_classes)}  classes')
        true_labeling = [unique_classes.index(a) for _, a in true_classes.values()]

        done(logger)

    # SAVE

    labelings = {
        'DominantSet'       : ds_labeling,
        'GaussianMixture'   : gmm_labeling,
        'NormalizedCut'     : nc_labeling,
        'Adjacent'          : trivial_labeling,
        'Random'            : random_labeling,
    }
    
    if TRUE_CLASSES: labelings['True'] = true_labeling

    labeling_fp = os.path.join(out_dir, 'labelings.npz')
    logger.info(mess=f'Saving labelings to {labeling_fp}')
    np.savez(labeling_fp, **labelings)

    logger.close()


