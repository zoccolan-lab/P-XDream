

from os import path
import numpy as np
from numpy.typing import NDArray

from analysis.utils.settings import ALEXNET_DIR, LAYER_SETTINGS
from pxdream.clustering.algo import (
    DominantSetClusteringAlgorithm,
    NormalizedCutClusteringAlgorithm,
    GaussianMixtureModelsClusteringAlgorithm,
    DBSCANClusteringAlgorithm
)
from pxdream.clustering.cluster import Clusters
from pxdream.clustering.model import PairwiseSimilarity
from pxdream.utils.logger     import Logger, LoguruLogger, SilentLogger

# --- SETTINGS ---

LAYER   = 'fc6-relu'

n_clu                   = 0                     # GMM, NC, ADJ, RAND
MIN_ELEMENTS            = 2                     # DS
MAX_ITER                = 50000                 # DS
FM_SIZE                 = 256                   # FM
EPS                     = 100                   # DBSCAN
MIN_SAMPLES             = 2                     # DBSCAN

GMM_DIM_REDUCTION    = {'type': 'pca', 'n_components': 500                                  } # GMM
DBSCAN_DIM_REDUCTION = {'type': 'tsne','n_components':   2, 'perplexity': 30, 'n_iter': 5000} # DBSCAN


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

def ds(data: NDArray, clu_dir: str, n_clu: int, logger: Logger = SilentLogger(),):
    
    algo = DominantSetClusteringAlgorithm(
        aff_mat=PairwiseSimilarity.cosine_similarity(data),
        min_elements=MIN_ELEMENTS,
        max_iter=MAX_ITER,
        logger=logger
    )
    
    clusters = algo.run()
    clusters.dump(clu_dir, logger=logger)

def gmm(data: NDArray, clu_dir: str, n_clu: int, logger: Logger = SilentLogger(),):
    
    algo = GaussianMixtureModelsClusteringAlgorithm(
        data=data,
        n_clusters=n_clu,
        dim_reduction=GMM_DIM_REDUCTION,
        logger=logger
    )
    
    clusters = algo.run()
    clusters.dump(clu_dir, logger=logger)
    
def nc(data: NDArray, clu_dir: str, n_clu: int, logger: Logger = SilentLogger(),):
    
    algo = NormalizedCutClusteringAlgorithm(
        aff_mat=PairwiseSimilarity.cosine_similarity(data),
        n_clusters=n_clu,
        logger=logger
    )
    
    clusters = algo.run()
    clusters.dump(clu_dir, logger=logger)

def dbscan(data: NDArray, clu_dir: str, n_clu: int, logger: Logger = SilentLogger(),):
    
    algo = DBSCANClusteringAlgorithm(
        data=data,
        eps=EPS,
        min_samples=MIN_SAMPLES,
        dim_reduction=DBSCAN_DIM_REDUCTION,
        logger=logger
    )
    
    clusters = algo.run()
    clusters.dump(clu_dir, logger=logger)
    
def adj(data: NDArray, clu_dir: str, n_clu: int, logger: Logger = SilentLogger(),):
    
    elements, *_ = data.shape
    
    logger.info(msg=f'Creating adjacent clusters with {n_clu} clusters')
    
    clusters = Clusters.adjacent_clusters(n_clu=n_clu, elements=elements)
    
    clusters.dump(clu_dir, logger=logger)

def rand(data: NDArray, clu_dir: str, n_clu: int, logger: Logger = SilentLogger(),):
    
    elements, *_ = data.shape
    
    logger.info(msg=f'Creating random clusters with {n_clu} clusters')
    
    clusters = Clusters.random_clusters(n_clu=n_clu, elements=elements)
    
    clusters.dump(clu_dir, logger=logger)

def fm(data: NDArray, clu_dir: str, n_clu: int, logger: Logger = SilentLogger(),):
    
    elements, *_ = data.shape
    
    logger.info(msg=f'Creating feature map clusters with {FM_SIZE} clusters')
    
    clusters = Clusters.adjacent_clusters(n_clu=FM_SIZE, elements=elements)
    setattr(clusters, 'NAME', "FeatureMapClusters")
    
    clusters.dump(clu_dir, logger=logger)


def main():
    
    logger = LoguruLogger(to_file=False)

    # Use the number of clusters from the DominantSet Clustering
    clu_dir  = path.join(ALEXNET_DIR, LAYER_SETTINGS[LAYER]['directory'], 'clusters')
    if path.exists(path.join(ALEXNET_DIR, LAYER_SETTINGS[LAYER]['directory'])):
        n_clu = len(Clusters.from_file(path.join(clu_dir, 'DominantSet.json')).clusters)
    
    data_fp = path.join(clu_dir, 'recordings.npy')
    logger.info(msg=f'Loading data from {data_fp}')
    data = np.load(data_fp)
    
    logger.info(msg='')
    logger.info(msg=f'Running clustering algorithms for {LAYER}')
    logger.info(msg='')
    
    # --- DS ---
    if CLU_ALGOS['ds']:
        logger.info(msg='Running Dominant Set Clustering')
        ds(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(msg='Skipping Dominant Set Clustering')
    logger.info(msg='')
    
    # --- GMM ---
    if CLU_ALGOS['gmm']:
        logger.info(msg='Running Gaussian Mixture Models Clustering')
        gmm(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(msg='Skipping Gaussian Mixture Models Clustering')
    logger.info(msg='')
    
    # --- NC ---
    if CLU_ALGOS['nc']:
        logger.info(msg='Running Normalized Cut Clustering')
        nc(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(msg='Skipping Normalized Cut Clustering')
    logger.info(msg='')
    
    # --- DBSCAN ---
    if CLU_ALGOS['dbscan']:
        logger.info(msg='Running DBSCAN Clustering')
        dbscan(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(msg='Skipping DBSCAN Clustering')
    logger.info(msg='')
    
    # --- ADJ ---
    if CLU_ALGOS['adj']:
        logger.info(msg='Running Adjacent Clustering')
        adj(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(msg='Skipping Adjacent Clustering')
    logger.info(msg='')
    
    # --- RAND ---
    if CLU_ALGOS['rand']:
        logger.info(msg='Running Random Clustering')
        rand(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(msg='Skipping Random Clustering')
    logger.info(msg='')
    
    # --- FM ---
    if CLU_ALGOS['fm'] and LAYER_SETTINGS[LAYER]['feature_map']:
        logger.info(msg='Running Feature Map Clustering')
        fm(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(msg='Skipping Feature Map Clustering')
    
    logger.info(msg='')
    
    logger.info(msg='Clustering algorithms finished')
    
if __name__ == '__main__':
    
    for layer in LAYER_SETTINGS:

        LAYER = layer
        
        main()
