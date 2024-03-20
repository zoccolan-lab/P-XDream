from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from os import path
from statistics import mean
from typing import Any, Counter, Dict, Iterable, List, Tuple, cast
import numpy as np
from numpy.typing import NDArray

from zdream.clustering.model import AffinityMatrix, Label, Labels
from zdream.logger import Logger, MutedLogger
from zdream.utils.io_ import read_json, save_json
from zdream.utils.misc import default

class DSCluster:
    ''' 
    Class representing a cluster result of Dominant Set clustering
    It collects a list of object identified by a label and their rank inside the group
    and a scalar representing the total internal coherence of the group.
    '''
    
    @dataclass
    class DSObject:
        '''
        Subclass representing an object inside the DS cluster
        with two attributes: 
        - the label identifying the object
        - the rank inside the cluster.
        '''
        
        label : Label
        rank  : np.float32
        
        def __str__ (self) -> str: return f'({self.label}, {self.rank})'
        def __repr__(self) -> str: return str(self)
        
        @property
        def info(self) -> Tuple[Label, np.float32]: 
            return self.label, self.rank
        
    def __init__(
        self, 
        labels: Labels, 
        ranks: NDArray[np.float32],
        w: np.float32
    ):
        '''
        Instantiate a new cluster 

        :param labels: Cluster object labels.
        :type labels: Labels
        :param ranks: Cluster object ranks.
        :type ranks: NDArray[np.float32]
        :param w: Cluster coherence score.
        :type w: np.float32
        '''
        
        self._w       = w
        self._objects = [self.DSObject(label=label, rank=rank) for label, rank in zip(labels, ranks)]
        
    # --- MAGIC METHODS ---
    
    def __str__ (self) -> str:                return f'DSCluster[objects: {len(self)}, coherence: {self.W}]'
    def __repr__(self) -> str:                return str(self)
    def __len__ (self) -> int:                return len (self.objects)
    def __iter__(self) -> Iterable[DSObject]: return iter(self.objects)
    
    # --- PROPERTIES ---
    
    @property
    def W(self): return self._w
    
    @property
    def objects(self): return self._objects
    
    # --- UTILS ---
    
    @staticmethod
    def extract_singletons(aff_mat: AffinityMatrix) -> List[DSCluster]:
        '''
        Extract trivial clusters with one element from an Affinity matrix.
        Their score and coherence is 1. 

        :return: List of singleton clusters clusters 
        :rtype: List[DSCluster]
        '''
        
        # NOTE: In the base case of the inductive definition
        #       of w_S(i) the base case corresponds to 1 when |S| = 1
        return [
            DSCluster(
                labels = np.array([lbl]), 
                ranks  = np.array([1.]),
                w      = np.float32(1.)
            )
            for lbl in aff_mat.labels
        ]

class DSClusters:
    '''
    Class for manipulating a set of `DSCluster` objects
    extracted from the same learning algorithm.
    
    The class compute some statistics on clusters and 
    allow to dump them to file.
    '''
    
    def __init__(self) -> None:
        '''
        Create an empty set of clusters
        '''
        
        self.empty()

    
    # --- MAGIC METHODS ---   
    
    def __str__ (self) -> str:                 return   f'DSClusters[n-clusters: {len(self)}, '\
                                                        f'avg per cluster: {round(self.avg_count, 3)}]'
    def __repr__(self) -> str:                 return str(self)
    def __len__ (self) -> int:                 return len (self.clusters)
    def __iter__(self) -> Iterable[DSCluster]: return iter(self.clusters)
    
    # --- PROPERTIES ---
    
    @property
    def clusters(self) -> List[DSCluster]: return self._clusters
    
    @property
    def clusters_counts(self) -> Dict[int, int]:
        '''
        Return a dictionary mapping the cluster cardinality
        to the number of clusters with that number of elements

        :return: Cluster cardinality mapping.
        :rtype: Dict[int, int]
        '''
        
        lens = [len(cluster) for cluster in self] # type: ignore
        return dict(Counter(lens))
    
    @property
    def avg_count(self) -> float:
        '''
        Return the average number of elements in 
        the cluster collection

        :return: Cluster cardinality mapping.
        :rtype: Dict[int, int]
        '''
        
        return sum([elements * count for elements, count in self.clusters_counts.items()]) / len(self)
    
    # --- UTILITIES ---
    
    def empty(self):
        '''
        Empty the set of clusters
        '''
        
        self._clusters: List[DSCluster] = []
        
    def add(self, cluster: DSCluster):
        '''
        Add a new cluster to the connection
        '''
        
        self._clusters.append(cluster)
        
    def dump(self, out_fp: str, logger: Logger | None = None):
        '''
        Store cluster information to file as a .JSON file

        :param out_fp: Output file path where to save clustering information
        :type out_fp: str
        :param logger: Logger to log i/o operation. If not given MutedLogger is set.
        :type logger: Logger | None, optional
        '''
        
        def obj_to_dict(obj: DSCluster.DSObject) -> Dict[str, Any]:
            return {
                'label': obj.label.tolist(),
                'rank' : obj. rank.tolist()
            }
        
        logger = default(logger, MutedLogger())
        
        out_dict = {
            f'DS_{i}': {
                'objects': [obj_to_dict(obj) for obj in cluster],
                "W": cluster.W.tolist()
            }
            for i, cluster in enumerate(self) # type: ignore
        } 
        
        fp = path.join(out_fp, 'DSClusters.json')
        logger.info(f'Saving clusters info to {fp}')
        save_json(data=out_dict, path=fp)
        
    @classmethod
    def from_file(cls, fp: str) -> DSClusters:
        '''
        Load a DSCluster state from JSON file

        :param fp: File path to .JSON file where DSClusters information is stored.
        :type fp: str
        :return: Loaded clusters object.
        :rtype: DSClusters
        '''
        
        data = read_json(path=fp)
        
        clusters = DSClusters()
        
        for cluster in data.values():
            
            clusters.add(
                cluster = DSCluster(
                    labels = np.array([obj['label'] for obj in cluster['objects']]),
                    ranks  = np.array([obj['rank' ] for obj in cluster['objects']]),
                    w      = np.float32(cluster['W'])
                )
            )
            
        return clusters
    
    
class DSClustering(ABC):
    '''
    Abstract class for performing generic DominantSet clustering
    using an affinity matrix.
    
    The class implements the constrained version of dominant set
    and has a `run` abstract method to be implemented in subclasses.
    
    See: Dominant Sets and Pairwise Clustering', 
         by Massimiliano Pavan and Marcello Pelillo, PAMI 2007.
    '''
        
    def __init__(
        self,
        aff_mat: AffinityMatrix,
        min_elements: int = 1,
        max_iter: int = 1000,
        delta_eps: float = 1e-8,
        zero_eps: float = 1e-12,
        logger: Logger | None = None
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
        :param logger: Logger to log clustering progress. If not given MutedLogger is set.
        :type logger: Logger | None, optional
        '''

        # Affinity Matrix
        self._aff_mat = aff_mat
        
        # Clustering hyperparameters
        self._min_elements = min_elements
        self._max_iter     = max_iter
        self._delta_eps    = delta_eps
        self._zero_eps     = zero_eps
        
        # Default logger
        self._logger = default(logger, MutedLogger())
        
        # Clusters
        self._clusters: DSClusters = DSClusters()
        
    # --- MAGIC METHODS ---
    
    def __str__(self) -> str: 
        return  f'DSClustering[objects: {len(self._aff_mat)}; '\
                f'min_elements: {self._min_elements}]'

    def __repr__(self) -> str: return str(self)
    
    # --- PROPERTIES ---
    
    @property
    def clusters(self) -> DSClusters:
        '''
        Returns clusters found by DS algorithm.
        It raises an error if not computed yet.
        '''
        
        if self._clusters:
            return self._clusters
        
        raise ValueError('Cluster not computed yet')
    
    # --- CLUSTERING ALGORITHM ---
    
    @staticmethod
    def _replicator_dynamics(
            aff_mat: AffinityMatrix,
            x: NDArray | None = None, # type: ignore
            alpha: float = 0.,
            max_iter: int = 1000,
            delta_eps: float = 1e-8,
            zero_eps: float = 1e-12,
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
            :param zero_eps: Approximation bound for probabilities zero flattening, defaults to 1e-12.
            :type zero_eps: float, optional
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
            a = aff_mat.aff_mat

            # Loop until convergence threshold
            # or until predetermined number of iterations            
            while iter < max_iter and dist > delta_eps:
                
                # Store old distribution for convergence comparison
                x_old = np.copy(x)  

                # Promote cluster membership with higher payoffs
                # x_i = x_i^T (Ax_i - \alpha x_i)
                x_ = x * (a.dot(x) - alpha * x)
                
                # Compute cluster coherency
                #w = (x * a.dot(x)).sum()

                # Compute normalization 
                # W_S = \sum_{i \in S} w_S(i)
                # den = x^T (A - \alpha I) x
                a_: NDArray = a - alpha * np.eye(len(aff_mat))
                den = (x * a_.dot(x)).sum()
            
                # Normalize the distribution
                x = x_ / den
                
                # Compute the change in distribution with previous iterations
                dist = np.sqrt(np.sum((x - x_old) ** 2))

                # Increment the iteration counter
                iter += 1
            
            # W_S = \sum_{i \in S} w_S(i)
            # w = den # TODO is this correct of should we compute w = (x * a.dot(x)).sum()
                    # before normalization? - this would add overhead at every cycle
                    
            x.clip(min=0)
                
            # Convergence flag
            converged = iter < max_iter
                
            return x, den, converged
    
    @abstractmethod
    def run(self):
        '''
        Run DS clustering algorithm with specified class hyper-parameters
        '''

        pass
    
    
class BaseDSClustering(DSClustering):
    
    def __init__(
        self, 
        aff_mat: AffinityMatrix, 
        min_elements: int = 1, 
        max_iter: int = 1000, 
        delta_eps: float = 1e-8, 
        zero_eps: float = 1e-12, 
        logger: Logger | None = None
    ) -> None:
        
        super().__init__(aff_mat, min_elements, max_iter, delta_eps, zero_eps, logger)
        
    @staticmethod
    def _replicator_dynamics(
        aff_mat: AffinityMatrix,
        x: NDArray | None = None, # type: ignore
        max_iter: int = 1000,
        delta_eps: float = 1e-8,
        zero_eps: float = 1e-12
    ) -> Tuple[NDArray, np.float32, bool]:
        
        # NOTE: The best-practice solution should be to implement a STUB
        #       to the replicator dynamic of the previous function defaulting
        #       hyperparameter alpha to 0. i.e. no spurious solutions.
        
        #       However this implies making twice the same dot product, 
        #       we reimplement the replicator dynamic in its simpler
        #       version to save a factor of 2.
        
        # NOTE: Use this if not interested in computational overhead
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
        a = aff_mat.aff_mat

        # Loop until convergence threshold
        # or until predetermined number of iterations            
        while iter < max_iter and dist > delta_eps:
            
            # Store old distribution for convergence comparison
            x_old = np.copy(x)  

            # Promote cluster membership with higher payoffs
            # x_i = x_i^T A x_i 
            x_ = x * a.dot(x)

            # Compute internal coherence 
            # W_S = x^T A x
            w = x_.sum()
        
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
        
    def run(self):
        
        # NOTE: The matrix is subject to subsetting during the
        #       clustering algorithm so we use a copy to prevent
        #       the original one from modifications
        aff_mat = copy(self._aff_mat)

        # NOTE: If the algorithm was already run, 
        #       all previous results are lost.
        self._clusters.empty()

        # Repeat until affinity matrix has positive sum of similarities
        # TODO: Check why is this condition
        
        iter = 1  # only used for logging
    
        while np.sum(aff_mat.aff_mat) > 0:

            # Perform 1-step of replicator dynamics
            x, w, converged = BaseDSClustering._replicator_dynamics(
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
                self._clusters.add(cluster=ds_cluster)
                
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
                self._clusters.add(cluster=singleton)


class HierarchicalDSClustering(DSClustering):

    def __init__(
        self, 
        aff_mat: AffinityMatrix, 
        min_elements: int = 1, 
        max_iter: int = 1000, 
        delta_eps: float = 1e-8, 
        zero_eps: float = 1e-12, 
        logger: Logger | None = None
    ) -> None:
        
        super().__init__(aff_mat, min_elements, max_iter, delta_eps, zero_eps, logger)
    
