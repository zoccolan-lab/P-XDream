

from os import path
import numpy as np
from numpy.typing import NDArray

from analysis.utils.settings import ALEXNET_DIR, LAYER_SETTINGS
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
    
    logger.info(mess=f'Creating adjacent clusters with {n_clu} clusters')
    
    clusters = Clusters.adjacent_clusters(n_clu=n_clu, elements=elements)
    
    clusters.dump(clu_dir, logger=logger)

def rand(data: NDArray, clu_dir: str, n_clu: int, logger: Logger = SilentLogger(),):
    
    elements, *_ = data.shape
    
    logger.info(mess=f'Creating random clusters with {n_clu} clusters')
    
    clusters = Clusters.random_clusters(n_clu=n_clu, elements=elements)
    
    clusters.dump(clu_dir, logger=logger)

def fm(data: NDArray, clu_dir: str, n_clu: int, logger: Logger = SilentLogger(),):
    
    elements, *_ = data.shape
    
    logger.info(mess=f'Creating feature map clusters with {FM_SIZE} clusters')
    
    clusters = Clusters.adjacent_clusters(n_clu=FM_SIZE, elements=elements)
    setattr(clusters, 'NAME', "FeatureMapClusters")
    
    clusters.dump(clu_dir, logger=logger)


def main():
    
    logger = LoguruLogger(on_file=False)

    # Use the number of clusters from the DominantSet Clustering
    clu_dir  = path.join(ALEXNET_DIR, LAYER_SETTINGS[LAYER]['directory'], 'clusters')
    if path.exists(path.join(ALEXNET_DIR, LAYER_SETTINGS[LAYER]['directory'])):
        n_clu = len(Clusters.from_file(path.join(clu_dir, 'DominantSet.json')).clusters)
    
    data_fp = path.join(clu_dir, 'recordings.npy')
    logger.info(mess=f'Loading data from {data_fp}')
    data = np.load(data_fp)
    
    logger.info(mess='')
    logger.info(mess=f'Running clustering algorithms for {LAYER}')
    logger.info(mess='')
    
    # --- DS ---
    if CLU_ALGOS['ds']:
        logger.info(mess='Running Dominant Set Clustering')
        ds(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(mess='Skipping Dominant Set Clustering')
    logger.info(mess='')
    
    # --- GMM ---
    if CLU_ALGOS['gmm']:
        logger.info(mess='Running Gaussian Mixture Models Clustering')
        gmm(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(mess='Skipping Gaussian Mixture Models Clustering')
    logger.info(mess='')
    
    # --- NC ---
    if CLU_ALGOS['nc']:
        logger.info(mess='Running Normalized Cut Clustering')
        nc(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(mess='Skipping Normalized Cut Clustering')
    logger.info(mess='')
    
    # --- DBSCAN ---
    if CLU_ALGOS['dbscan']:
        logger.info(mess='Running DBSCAN Clustering')
        dbscan(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(mess='Skipping DBSCAN Clustering')
    logger.info(mess='')
    
    # --- ADJ ---
    if CLU_ALGOS['adj']:
        logger.info(mess='Running Adjacent Clustering')
        adj(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(mess='Skipping Adjacent Clustering')
    logger.info(mess='')
    
    # --- RAND ---
    if CLU_ALGOS['rand']:
        logger.info(mess='Running Random Clustering')
        rand(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(mess='Skipping Random Clustering')
    logger.info(mess='')
    
    # --- FM ---
    if CLU_ALGOS['fm'] and LAYER_SETTINGS[LAYER]['feature_map']:
        logger.info(mess='Running Feature Map Clustering')
        fm(data=data, clu_dir=clu_dir, n_clu=n_clu, logger=logger)
    else: logger.info(mess='Skipping Feature Map Clustering')
    
    logger.info(mess='')
    
    logger.info(mess='Clustering algorithms finished')
    
if __name__ == '__main__':
    
    for layer in LAYER_SETTINGS:

        LAYER = layer
        
        main()
