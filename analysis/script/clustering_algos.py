

from os import path
import numpy as np
from numpy.typing import NDArray

from analysis.utils.settings import CLUSTER_DIR, LAYER_SETTINGS
from zdream.clustering.algo import (
    DominantSetClusteringAlgorithm,
    NormalizedCutClusteringAlgorithm,
    GaussianMixtureModelsClusteringAlgorithm,
    DBSCANClusteringAlgorithm
)
from zdream.clustering.cluster import Clusters
from zdream.clustering.model import PairwiseSimilarity
from zdream.utils.logger     import Logger, LoguruLogger, SilentLogger

# --- SETTINGS ---

LAYER   = 'fc6-relu'

CLU_DIR  = path.join(CLUSTER_DIR, LAYER_SETTINGS[LAYER]['directory'])

N_CLU                   = 0                     # GMM, NC, ADJ, RAND
MIN_ELEMENTS            = 2                     # DS
MAX_ITER                = 50000                 # DS
FM_SIZE                 = 256                   # FM
EPS                     = 100                   # DBSCAN
MIN_SAMPLES             = 2                     # DBSCAN

GMM_DIM_REDUCTION    = {'type': 'pca', 'n_components': 500                                  } # GMM
DBSCAN_DIM_REDUCTION = {'type': 'tsne','n_components':   2, 'perplexity': 30, 'n_iter': 5000} # DBSCAN

# Use the number of clusters from the DominantSet Clustering
if path.exists(path.join(CLUSTER_DIR, LAYER_SETTINGS[LAYER]['directory'])):
    N_CLU = len(Clusters.from_file(path.join(CLU_DIR, 'DominantSetClusters.json')).clusters)

# ALGOS FLAGS

CLU_ALGOS = {
    'ds'    : False,
    'gmm'   : True,
    'nc'    : True,
    'dbscan': False,
    'adj'   : True,
    'rand'  : True,
    'fm'    : True
}


# --- RUN ---

def ds(data: NDArray, logger: Logger = SilentLogger()):
    
    algo = DominantSetClusteringAlgorithm(
        aff_mat=PairwiseSimilarity.cosine_similarity(data),
        min_elements=MIN_ELEMENTS,
        max_iter=MAX_ITER,
        logger=logger
    )
    
    clusters = algo.run()
    clusters.dump(CLU_DIR, logger=logger)

def gmm(data: NDArray, logger: Logger = SilentLogger()):
    
    algo = GaussianMixtureModelsClusteringAlgorithm(
        data=data,
        n_clusters=N_CLU,
        dim_reduction=GMM_DIM_REDUCTION,
        logger=logger
    )
    
    clusters = algo.run()
    clusters.dump(CLU_DIR, logger=logger)
    
def nc(data: NDArray, logger: Logger = SilentLogger()):
    
    algo = NormalizedCutClusteringAlgorithm(
        aff_mat=PairwiseSimilarity.cosine_similarity(data),
        n_clusters=N_CLU,
        logger=logger
    )
    
    clusters = algo.run()
    clusters.dump(CLU_DIR, logger=logger)

def dbscan(data: NDArray, logger: Logger = SilentLogger()):
    
    algo = DBSCANClusteringAlgorithm(
        data=data,
        eps=EPS,
        min_samples=MIN_SAMPLES,
        dim_reduction=DBSCAN_DIM_REDUCTION,
        logger=logger
    )
    
    clusters = algo.run()
    clusters.dump(CLU_DIR, logger=logger)
    
def adj(data: NDArray, logger: Logger = SilentLogger()):
    
    elements, *_ = data.shape
    
    logger.info(mess=f'Creating adjacent clusters with {N_CLU} clusters')
    
    clusters = Clusters.adjacent_clusters(n_clu=N_CLU, elements=elements)
    
    clusters.dump(CLU_DIR, logger=logger)

def rand(data: NDArray, logger: Logger = SilentLogger()):
    
    elements, *_ = data.shape
    
    logger.info(mess=f'Creating random clusters with {N_CLU} clusters')
    
    clusters = Clusters.random_clusters(n_clu=N_CLU, elements=elements)
    
    clusters.dump(CLU_DIR, logger=logger)

def fm(data: NDArray, logger: Logger = SilentLogger()):
    
    elements, *_ = data.shape
    
    logger.info(mess=f'Creating feature map clusters with {FM_SIZE} clusters')
    
    clusters = Clusters.adjacent_clusters(n_clu=FM_SIZE, elements=elements)
    setattr(clusters, 'NAME', "FeatureMapClusters")
    
    clusters.dump(CLU_DIR, logger=logger)


def main():
    
    logger = LoguruLogger(on_file=False)
    
    data_fp = path.join(CLU_DIR, 'recordings.npy')
    logger.info(mess=f'Loading data from {data_fp}')
    data = np.load(data_fp)
    
    logger.info(mess='')
    logger.info(mess=f'Running clustering algorithms for {LAYER}')
    logger.info(mess='')
    
    # --- DS ---
    if CLU_ALGOS['ds']:
        logger.info(mess='Running Dominant Set Clustering')
        ds(data, logger)
    else: logger.info(mess='Skipping Dominant Set Clustering')
    logger.info(mess='')
    
    # --- GMM ---
    if CLU_ALGOS['gmm']:
        logger.info(mess='Running Gaussian Mixture Models Clustering')
        gmm(data, logger)
    else: logger.info(mess='Skipping Gaussian Mixture Models Clustering')
    logger.info(mess='')
    
    # --- NC ---
    if CLU_ALGOS['nc']:
        logger.info(mess='Running Normalized Cut Clustering')
        nc(data, logger)
    else: logger.info(mess='Skipping Normalized Cut Clustering')
    logger.info(mess='')
    
    # --- DBSCAN ---
    if CLU_ALGOS['dbscan']:
        logger.info(mess='Running DBSCAN Clustering')
        dbscan(data, logger)
    else: logger.info(mess='Skipping DBSCAN Clustering')
    logger.info(mess='')
    
    # --- ADJ ---
    if CLU_ALGOS['adj']:
        logger.info(mess='Running Adjacent Clustering')
        adj(data, logger)
    else: logger.info(mess='Skipping Adjacent Clustering')
    logger.info(mess='')
    
    # --- RAND ---
    if CLU_ALGOS['rand']:
        logger.info(mess='Running Random Clustering')
        rand(data, logger)
    else: logger.info(mess='Skipping Random Clustering')
    logger.info(mess='')
    
    # --- FM ---
    if CLU_ALGOS['fm'] and LAYER_SETTINGS[LAYER]['feature_map']:
        logger.info(mess='Running Feature Map Clustering')
        fm(data, logger)
    else: logger.info(mess='Skipping Feature Map Clustering')
    
    logger.info(mess='')
    
    logger.info(mess='Clustering algorithms finished')
    
if __name__ == '__main__': main()
