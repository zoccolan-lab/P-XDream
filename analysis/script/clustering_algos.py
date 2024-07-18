

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
ds_n_clu = len(Clusters.from_file(path.join(CLU_DIR, 'DominantSetClusters.json')).clusters)

N_CLU                   = ds_n_clu
PCA_COMPONENTS_GMM      = 500               # GMM
PCA_COMPONENTS_DBSCAN   = 3                 # DBSCAN
MIN_ELEMENTS            = 2                 # DS
MAX_ITER                = 50000             # DS
FM_SIZE                 = 256               # FM
EPS_GRID_SEARCH         = [float(f) for f in range(1, 501, 10)]      # DBSCAN
MIN_SAMPLES_GRID_SEARCH = list(range(2, 11)) # DBSCAN

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
        n_components=PCA_COMPONENTS_GMM,
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
    
    algo1, algo2 = DBSCANClusteringAlgorithm.grid_search(
        data=data,
        eps=EPS_GRID_SEARCH,
        min_samples=MIN_SAMPLES_GRID_SEARCH,
        len_target=N_CLU,
        n_components=PCA_COMPONENTS_DBSCAN,
        logger=logger
    )
    
    clusters = algo1.run()
    clusters.__setattr__('NAME', 'DBSCANSilhouetteClusters')
    clusters.dump(CLU_DIR, logger=logger)
    
    if algo2 is not None:
        
        clusters = algo2.run()
        clusters.__setattr__('NAME', 'DBSCANDimTargetClusters')
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
