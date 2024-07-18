from copy import copy
import itertools

from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import torch
from zdream.clustering.cluster import Clusters
from zdream.clustering.ds import DSCluster, DSClusters
from zdream.clustering.model import AffinityMatrix
from zdream.utils.logger import Logger, SilentLogger
from zdream.utils.misc import default, device


import numpy as np
from numpy.typing import NDArray


from abc import ABC, abstractmethod
from typing import List, Tuple, Union, cast

def pca(data: NDArray, n_components: int | None, logger: Logger = SilentLogger()):
    
    # Perform PCA
    if n_components is None: return data
        
    logger.info(f'Performing PCA up to {n_components} components.')
    
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_new = pca.transform(data)
    
    return data_new

class ClusteringAlgorithm(ABC):
    '''
    Abstract class for clustering algorithms.
    '''
    
    def __init__(self) -> None:
        '''
        Create a new instance of ClusteringAlgorithm.
        '''
        
        self._run_flag  = False
        self._clusters = Clusters()
        
    def __str__ (self) -> str : return f'ClusteringAlgorithm[]'
    def __repr__(self) -> str : return str(self)
    def __bool__(self) -> bool: return self._run_flag
        
    def run(self) -> Clusters:
        '''
        Run the clustering algorithm and return the clusters found.
        The method saves the clusters in the internal state to avoid recomputation.
        '''
        
        self._clusters = self._run()
        self._run_flag = True
        
        return self.clusters

    @abstractmethod
    def _run(self) -> Clusters: pass
    ''' Run the clustering algorithm. '''

    
    @property
    def clusters(self) -> Clusters:
        '''
        Returns clusters found by run algorithm.
        It raises an error if not computed yet.
        '''

        if self: return self._clusters

        raise ValueError('Cluster not computed yet')


class GaussianMixtureModelsClusteringAlgorithm(ClusteringAlgorithm):
    '''
    Class for performing Gaussian Mixture Model clustering
    by performing PCA and clustering on the reduced space.
    '''
    
    def __init__(
        self,
        data         : NDArray,
        n_clusters   : int,
        n_components : int | None = None,
        logger       : Logger     = SilentLogger()
    ) -> None:
        
        super().__init__()
        
        self._data         = data
        self._n_clusters   = n_clusters
        self._n_components = n_components
        self._logger       = logger
    
    def __str__(self) -> str:         
        
        return  f'GaussianMixtureModels{super().__str__()[:-1]}n-clusters: {len(self._clusters)}; '\
                f'n-components: {self._n_components if self._n_components else "All"}]'

    def _run(self) -> Clusters:
        
        # Perform PCA
        data = pca(data=self._data, n_components=self._n_components, logger=self._logger)
        
        # Perform GMM clustering
        self._logger.info(f'Performing GMM clustering with {self._n_clusters} clusters')
        
        gmm = GaussianMixture(n_components=self._n_clusters)
        labels = gmm.fit_predict(data)
        
        # Create clusters
        clusters = Clusters.from_labeling(labeling=labels)
        setattr(clusters, "NAME", "GaussianMixtureClusters")
        
        return clusters
        

class NormalizedCutClusteringAlgorithm(ClusteringAlgorithm):
    '''
    Class performing normalized cuts clustering on an affinity matrix.
    '''
    
    def __init__(
        self,
        aff_mat      : AffinityMatrix,
        n_clusters   : int,
        logger       : Logger = SilentLogger()
    ) -> None:
        
        super().__init__()
        
        self._aff_mat    = aff_mat
        self._n_clusters = n_clusters
        self._logger     = logger
    
    def __str__(self) -> str: return f'NormalizedCut{super().__str__()[:-1]}n-clusters: {len(self._clusters)}]'
        
    def _run(self) -> Clusters:
        
        self._logger.info(f'Performing Normalized Cuts clustering with {self._n_clusters} clusters')
        
        # Perform spectral clustering
        spectral = SpectralClustering(n_clusters=self._n_clusters, affinity='precomputed')
        labels   = spectral.fit_predict(self._aff_mat.A)

        # Create clusters
        clusters = Clusters.from_labeling(labeling=labels)
        setattr(clusters, "NAME", "NormalizedCutClusters")
        
        return clusters        

class DBSCANClusteringAlgorithm(ClusteringAlgorithm):
    '''
    Class performing normalized cuts clustering on an affinity matrix.
    '''
    
    def __init__(
        self,
        data         : NDArray,
        eps          : float,
        min_samples  : int,
        n_components : int | None = None,
        logger       : Logger     = SilentLogger()
    ) -> None:
        
        super().__init__()
        
        self._data         = data
        self._eps          = eps
        self._min_samples  = min_samples
        self._n_components = n_components
        self._logger       = logger
    
    def __str__(self) -> str: return    f'DBSCAN{super().__str__()[:-1]}eps: {self._eps}; '\
                                        f'min_samples: {self._min_samples}; '\
                                        f'n-components: {self._n_components if self._n_components else "All"}]'
        
    def _run(self) -> Clusters:
        
        # Perform PCA
        data = pca(data=self._data, n_components=self._n_components, logger=self._logger)
        
        self._logger.info(f'Performing DBSCAN clustering with parameters: eps={self._eps}, min_samples={self._min_samples}')
        
        # Perform spectral clustering
        dbscan_algo = DBSCAN(eps=self._eps, min_samples=self._min_samples, metric='euclidean', n_jobs=-1)
        dbscan_algo.fit(data)
        labels = dbscan_algo.labels_.astype(int)

        # Create clusters
        clusters = Clusters.from_labeling(labeling=labels)
        setattr(clusters, "NAME", "DBSCANClusters")
        
        return clusters
    
    @classmethod
    def grid_search(
            cls, 
            data         : NDArray, 
            eps          : List[float], 
            min_samples  : List[int],
            len_target   : int | None = None,
            n_components : int | None = None,
            logger       : Logger = SilentLogger()
    ) -> Tuple['DBSCANClusteringAlgorithm', Union['DBSCANClusteringAlgorithm', None]]:
        
        data_pca = pca(data=data, n_components=n_components, logger=logger)

        tpl = list(itertools.product(eps, min_samples))
        
        if len(tpl) == 0: raise ValueError('Empty grid search parameters')
        
        logger.formatting = lambda x: f' > {x}'
        
        best_silhouette = -1
        best_tpl = (-1., -1)
        
        if len_target is not None:
            best_len = float('inf')
            best_tpl2 = (-1., -1)
        
        logger.info('Performing DBSCAN grid search')
        
        for eps_, min_samples_ in tpl:
            
            algo = cls(data=data_pca, eps=eps_, min_samples=min_samples_, logger=logger)
            algo.run()
            
            sil = algo.clusters.silhouette_score(data=data_pca)
                
            logger.info(f' > Silhouette coefficient={sil}')
            
            if sil > best_silhouette:
                best_silhouette = sil
                best_tpl = (eps_, min_samples_)
            
            if len_target is not None:
                
                clu_dist = abs(len(algo.clusters) - len_target)
                
                logger.info(f' > Cluster found={len(algo.clusters)}')
                
                if clu_dist < best_len:
                    best_len = clu_dist
                    best_tpl2 = (eps_, min_samples_)
                

        logger.reset_formatting()
        
        best_esp,  best_min_samples  = best_tpl
        
        logger.info(f'Selected clustering algorithm with eps={best_esp}, min_samples={best_min_samples} with silhouette coeff {best_silhouette}.')
        
        if len_target is not None:
            best_esp2, best_min_samples2 = best_tpl2
            logger.info(f'Selected clustering algorithm with eps={best_esp2}, min_samples={best_min_samples2} with cluster size {abs(best_len - len_target)}.')
        
        return DBSCANClusteringAlgorithm(data=data, eps=best_esp,  min_samples=best_min_samples,  n_components=n_components, logger=logger),\
               DBSCANClusteringAlgorithm(data=data, eps=best_esp2, min_samples=best_min_samples2, n_components=n_components, logger=logger) if len_target is not None else None


class DominantSetClusteringAlgorithm(ClusteringAlgorithm):
    '''
    Abstract class for performing generic DominantSet clustering
    using an affinity matrix.

    The class implements the constrained version of dominant set
    and has a `run` abstract method to be implemented in subclasses.

    SEE:    Dominant Sets and Pairwise Clustering', 
            by Massimiliano Pavan and Marcello Pelillo, PAMI 2007.
    '''

    def __init__(
        self,
        aff_mat      : AffinityMatrix,
        min_elements : int = 1,
        max_iter     : int = 1000,
        delta_eps    : float = 1e-8,
        zero_eps     : float = 1e-12,
        logger       : Logger = SilentLogger()
    ) -> None:
        '''
        Create a new instance of DSClustering given its affinity matrix
        and setting its hyperparameters.

        :param aff_mat: Affinity matrix containing object pairwise similarities.
        :type aff_mat: AffinityMatrix
        :param min_elements: Minimum number of elements in a cluster, defaults to 1.
        :type min_elements: int, optional
        :param max_iter: Maximum number of iterations of the replicator dynamics, defaults to 1000.
        :type max_iter: int, optional
        :param delta_eps: Convergence threshold for the replicator dynamics.
            Maximum difference between two consecutive strategies to have convergence, defaults to 1e-8.
        :type delta_eps: float, optional
        :param zero_eps: Approximation bound for probabilities zero flattening, defaults to 1e-12.
        :type zero_eps: float, optional
        :param logger: Logger to log clustering progress. If not given SilentLogger is set.
        :type logger: Logger | None, optional
        '''
        
        super().__init__()
        
        self._clusters = DSClusters()  # Use DSClusters to store clusters

        # Affinity Matrix
        self._aff_mat      = aff_mat

        # Clustering hyperparameters
        self._min_elements = min_elements
        self._max_iter     = max_iter
        self._delta_eps    = delta_eps
        self._zero_eps     = zero_eps

        # Default logger
        self._logger = logger

    # --- STRING REPRESENTATION ---

    def __str__(self) -> str:
        return  f'DominantSet{super().__str__()[:-1]}max_iter: {self._max_iter}; '\
                f'min_elements: {self._min_elements}; '\
                f'delta_eps: {self._delta_eps}; '\
                f'zero_eps: {self._zero_eps}]'

    # --- CLUSTERING ALGORITHM ---

    def _replicator_dynamics(
        self,
        aff_mat   : AffinityMatrix,
        x         : NDArray | None = None, # type: ignore
        alpha     : float          = 0.,
        max_iter  : int            = 1000,
        delta_eps : float          = 1e-8
    ) -> Tuple[NDArray, np.float32, bool]:
        '''
        Perform one-step replicator dynamics on the given affinity matrix.

        Replicator dynamics is an algorithm used for clustering based on affinity matrix.
        It iteratively updates the distribution of cluster memberships until convergence or
        maximum iteration limit is reached.

        :param aff_mat: Affinity matrix representing the pairwise similarities between data points.
        :param aff_mat: Initial distribution of cluster memberships. If not provided, a uniform
            distribution will be used.
        :param alpha: Regularization factor for spurious solutions.
        :type alpha: float
        :param max_iter: Maximum number of iterations of the replicator dynamics, defaults to 1000.
        :type max_iter: int, optional
        :param delta_eps: Convergence threshold for the replicator dynamics.
            Maximum difference between two consecutive strategies to have convergence, defaults to 1e-8.
        :type delta_eps: float, optional
        :return: Tuple containing the final distribution of cluster memberships, their coherence and 
            a boolean indicating if the process converged.
        '''

        # In the case initial distribution is not given use a uniform one
        x: NDArray = default(x, np.zeros(len(aff_mat)) + 1 / len(aff_mat))

        # Initialize hyperparameters
        dist:  float = 2 * delta_eps  # this is just for entering the first loop
        iter:  int   = 0

        # We extract the actual matrix from the object
        # to use it's method and avoid computational overhead of
        # class internal checks
        a = aff_mat.A

        # Loop until convergence threshold
        # or until predetermined number of iterations            
        while iter < max_iter and dist > delta_eps:

            # Store old distribution for convergence comparison
            x_old = np.copy(x)

            # Promote cluster membership with higher payoffs
            # $ x_i = x_i^T (Ax_i - \alpha x_i) $
            x_ = x * (a.dot(x) - alpha * x)

            # Compute cluster coherency
            # $ w = (x * a.dot(x)).sum() $

            # Compute normalization 
            # $ W_S = \sum_{i \in S} w_S(i) $
            # $ den = x^T (A - \alpha I) x  $
            a_: NDArray = a - alpha * np.eye(len(aff_mat))
            den = (x * a_.dot(x)).sum()

            # Normalize the distribution
            x = x_ / den

            # Compute the change in distribution with previous iterations
            dist = np.sqrt(np.sum((x - x_old) ** 2))

            # Increment the iteration counter
            iter += 1

        x.clip(min=0)

        # Convergence flag
        converged = iter < max_iter

        return x, den, converged
    
    @property
    def clusters(self) -> DSClusters:
        '''
        Returns clusters found by run algorithm.
        It raises an error if not computed yet.
        '''

        return cast(DSClusters, super().clusters)

    def _run(self):
        '''
        Run DS clustering algorithm with specified class hyper-parameters
        '''
        
        err_msg = """
            The basic Dominant Set class is used to provide a common interface for replicator dynamics and hyperparameters tuning.
            The specific clustering algorithm should implement the run method in subclasses.
        """

        raise NotImplementedError(err_msg)


class BaseDSClustering(DominantSetClusteringAlgorithm):

    def __init__(
        self,
        aff_mat      : AffinityMatrix,
        min_elements : int    = 1,
        max_iter     : int    = 1000,
        delta_eps    : float  = 1e-8,
        zero_eps     : float  = 1e-12,
        logger       : Logger = SilentLogger()
    ) -> None:

        super().__init__(aff_mat, min_elements, max_iter, delta_eps, zero_eps, logger)

    def _replicator_dynamics(
        self,
        aff_mat   : AffinityMatrix,
        x         : NDArray | None = None, # type: ignore
        max_iter  : int            = 1000,
        delta_eps : float          = 1e-8,
    ) -> Tuple[NDArray, np.float32, bool]:

        # NOTE: The best-practice solution should be to implement a STUB
        #       to the replicator dynamic of the previous function defaulting
        #       hyperparameter alpha to 0. i.e. no spurious solutions.

        #       However this implies making twice the same dot product, 
        #       we reimplement the replicator dynamic in its simpler
        #       version to save a factor of 2.

        # NOTE: Use this if not interested in computational overhead
        #
        # return DSClustering._replicator_dynamics(
        #     aff_mat=aff_mat, 
        #     x=x,
        #     alpha=0, # this is the core of the logic
        #     max_iter=max_iter,
        #     delta_eps=delta_eps, 
        #     zero_eps=zero_eps
        # )

        # In the case initial distribution is not given use a uniform one
        x: NDArray = default(x, np.zeros(len(aff_mat)) + 1 / len(aff_mat))

        # Initialize hyperparameters
        dist:  float = 2 * delta_eps  # this is just for entering the first loop
        iter:  int   = 0

        # We extract the actual matrix from the object
        # to use it's method and avoid computational 
        # overhead of class internal checks
        a = aff_mat.A
        w = np.float32(0.)

        # Loop until convergence threshold
        # or until predetermined number of iterations            
        while iter < max_iter and dist > delta_eps:

            # Store old distribution for convergence comparison
            x_old = np.copy(x)

            # Promote cluster membership with higher payoffs
            # $ x_i = x_i^T A x_i $
            x_ = x * a.dot(x)

            # Compute internal coherence 
            # $ W_S = x^T A x $
            w_old = w
            w: np.float32 = x_.sum()
            
            delta_w = w - w_old
            if delta_w < 0:
                self._logger.warn(mess=f'Iteration {iter}: coherence decreased of {delta_w}. ')

            # Normalize the distribution
            x = x_ / w

            # Compute the change in distribution with previous iterations
            dist = np.sqrt(np.sum((x - x_old) ** 2))

            # Increment the iteration counter
            iter += 1

        x.clip(min=0)

        # Convergence flag
        converged = iter < max_iter

        return x, w, converged

    def _run(self) -> DSClusters:

        # NOTE: The matrix is subject to subsetting during the
        #       clustering algorithm so we use a copy to prevent
        #       the original one from modifications
        aff_mat = copy(self._aff_mat)

        clusters = DSClusters()

        # Repeat until affinity matrix has positive sum of similarities

        iter = 1  # only used for logging

        while aff_mat:
            
            info = f'Running clustering step {iter}. Elements to cluster: {len(aff_mat)}'
            self._logger.info(mess=info)

            # Perform 1-step of replicator dynamics
            x, w, converged = self._replicator_dynamics(
                aff_mat=aff_mat,
                max_iter=self._max_iter
            )

            # Warning if it didn't converge
            if not converged:
                wrn_msg = f'Cluster {iter} didn\'t converge in {self._max_iter} iterations. '
                self._logger.warn(wrn_msg)

            # Extract the cluster as vector support
            support: NDArray[np.int32] = np.where(x > self._zero_eps)[0]

            # Create the extracted cluster
            ds_cluster = DSCluster(
                labels=aff_mat.labels[support],
                ranks=x[support],
                w=w
            )

            # Add the cluster if it satisfies the minimum element constraint
            if len(ds_cluster) >= self._min_elements:
                clusters.add(cluster=ds_cluster)

            # Warn if the extracted cluster was not added because of minimum
            # element constraint
            else:
                wrn_msg =   f'Cluster with {len(ds_cluster)} elements found at iteration {iter}, '\
                            f'but it don\'t satisfy minimum number of elements requires which is {self._min_elements}.'
                self._logger.warn(wrn_msg)

            # Subset affinity matrix with non clustered objects            
            aff_mat.delete_objects(ids=support, inplace=True)

            iter += 1

        # In the case the minimum number of elements in a cluster is zero 
        # Add all remaining objects as singleton clusters
        if self._min_elements == 1:
            for singleton in DSCluster.extract_singletons(aff_mat=aff_mat):
                clusters.add(cluster=singleton)
        else:
            wrn_msg = f'Excluding from the clustering {len(aff_mat)} singleton_elements'
            self._logger.warn(wrn_msg)
        
        setattr(clusters, "NAME", "DominantSetClusters")

        return clusters

class BaseDSClusteringGPU(BaseDSClustering):
    '''
    Perform Dominant Set clustering using GPU acceleration.
    '''
    
    def __init__(
        self, 
        aff_mat      : AffinityMatrix,
        min_elements : int    = 1,
        max_iter     : int    = 1000,
        delta_eps    : float  = 1e-8,
        zero_eps     : float  = 1e-12,
        logger       : Logger = SilentLogger()
    ) -> None:
        super().__init__(aff_mat, min_elements, max_iter, delta_eps, zero_eps, logger)
    
    def _replicator_dynamics(
        self,
        aff_mat   : AffinityMatrix,
        x         : NDArray | None = None, # type: ignore
        max_iter  : int            = 1000,
        delta_eps : float          = 1e-8,
    ) -> Tuple[NDArray, np.float32, bool]:

        # NOTE: We replicate the same algorithm as the CPU version
        #       but we use the GPU accelerated version of the matrix
        #       multiplication by leveraging the torch library.

        # In the case initial distribution is not given use a uniform one
        x_ten = torch.tensor(
            default(x, np.zeros(len(aff_mat)) + 1 / len(aff_mat)),
            dtype=torch.float64,
            device=device
        )
        
        # Initialize hyperparameters
        dist = 2 * delta_eps  # this is just for entering the first loop
        iter = 0

        # We extract the actual matrix from the object
        # to use its method and avoid computational 
        # overhead of class internal checks
        a = torch.tensor(aff_mat.A, dtype=torch.float64, device=device)
        w = torch.tensor(0., dtype=torch.float64)

        # Loop until convergence threshold
        # or until predetermined number of iterations            
        while iter < max_iter and dist > delta_eps:

            # Store old distribution for convergence comparison
            x_old = x_ten.clone()

            # Promote cluster membership with higher payoffs
            x_ten_ = x_ten * torch.matmul(a, x_ten)

            # Compute internal coherence 
            # W_S = x^T A x
            w_old = w
            w = x_ten_.sum()
            
            delta_w = w - w_old
            if delta_w < 0:
                self._logger.warn(mess=f'Iteration {iter}: coherence decreased of {delta_w}. ')

            # Normalize the distribution
            x_ten = x_ten_.div(w)

            # Compute the change in distribution with previous iterations
            dist = torch.sqrt(torch.sum((x_ten - x_old) ** 2))

            # Increment the iteration counter
            iter += 1

        x_ten = x_ten.clamp(min=0)

        # Convergence flag
        converged = iter < max_iter

        return x_ten.cpu().numpy(), w.cpu().numpy(), converged

class HierarchicalDSClustering(DominantSetClusteringAlgorithm):
    
    def __init__(
        self, 
        aff_mat      : AffinityMatrix, 
        min_elements : int    = 1, 
        max_iter     : int    = 1000, 
        delta_eps    : float  = 1e-8,
        zero_eps     : float  = 1e-12, 
        logger       : Logger = SilentLogger()
    ) -> None:
        
        super().__init__(aff_mat, min_elements, max_iter, delta_eps, zero_eps, logger)
        
    
    def run(self):
        raise NotImplementedError('Not implemented yet')