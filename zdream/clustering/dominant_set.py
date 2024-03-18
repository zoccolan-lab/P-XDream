from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from os import path
from statistics import mean
from typing import Any, Counter, Dict, Iterable, List, Tuple, cast
import numpy as np
from numpy.typing import NDArray

from zdream.logger import Logger, MutedLogger
from zdream.utils.io_ import read_json, save_json
from zdream.utils.misc import default

Label  = np.int32 | np.str_
Labels = NDArray[Label]

class AffinityMatrix:
    '''
    Class representing a 2-dimensional square affinity matrix
    It allows to associate labels to each object and perform
    basic operations such as copying and subsetting
    '''
    
    def __init__(self, aff_mat: NDArray, labels: Labels | None = None) -> None:
        '''
        Instantiate a new object with affinity matrix
        It performs sanity check for shape and values

        :param aff_mat: Affinity matrix.
        :type aff_mat: NDArray
        :param labels: Labels associated to each object.
                       If not given matrix indexes are used.
        :type labels: Labels
        '''
        
        # Sanity checks
        
        # A) Shape check
        if len(aff_mat.shape) != 2 or aff_mat.shape[0] != aff_mat.shape[1]:
            err_msg = f'The affinity matrix is supposed to be square, but {aff_mat.shape} shape found.'
            raise ValueError(err_msg)
        
        # B) Symmetric check
        if not np.array_equal(aff_mat, aff_mat.T):
            err_msg = f'The affinity matrix it\'s not symmetric.'
            raise ValueError(err_msg)
        
        # C) Zero diagonal check
        if np.any(np.diag(aff_mat) != 0):
            err_msg = f'The affinity matrix diagonal contains non-zero values.'
            raise ValueError(err_msg)
            
        # D) Positive similarities 
        if np.any(aff_mat < 0):
            err_msg = f'The affinity matrix contains negative similarities.'
            raise ValueError(err_msg)
        
        # Save matrix and labels
        self._aff_mat: NDArray = aff_mat
        self._labels : Labels  = default(labels, np.array(list(range(len(self)))))
    
    # --- MAGIC METHODS ---
    
    def __len__ (self) -> int: return self.shape[0]
    def __str__ (self) -> str: return f'AffinityMatrix[objects: {len(self)}]'
    def __repr__(self) -> str: return str(self)
    
    def __copy__(self) -> 'AffinityMatrix': 
        return AffinityMatrix(
            aff_mat = np.copy(self.aff_mat), 
            labels  = np.copy(self.labels)
        )
    
    # --- PROPERTIES ---
        
    @property
    def aff_mat(self) -> NDArray: return self._aff_mat
    
    @property
    def labels(self) -> Labels: return self._labels
    
    @property
    def shape(self) -> Tuple[int, ...]: return self.aff_mat.shape
    
    # --- UTILITIES ---
        
    def delete_objects(self, ids: NDArray[np.int32], inplace: bool = False) -> AffinityMatrix | None:
        '''
        Performs a subsetting of the affinity matrix by deleting objects with specified indexes.
        The modification can be either inplace or generate a new instance.

        :param ids: Object ids to perform subsetting.
        :type ids: NDArray[np.int32]
        :param inplace: If to apply the subsetting to current object or if to generate 
                        a new one, defaults to False.
        :type inplace: bool, optional
        :return: The subsetted matrix if the operation is not inplace, None if it is inplace.
        :rtype: AffinityMatrix | None
        '''
        
        # Subset labels
        new_labels = np.delete(self._labels, ids)
            
        # Subset columns and rows
        new_mat = np.delete(self._aff_mat, ids, axis=0)
        new_mat = np.delete(      new_mat, ids, axis=1)
        
        # Return new instance if not inplace operation
        if not inplace:
            return AffinityMatrix(aff_mat=new_mat, labels=new_labels)
        
        # Update attributes of current objects
        self._labels  = new_labels
        self._aff_mat = new_mat
        
    def dot(self, arr: NDArray) -> NDArray:
        '''
        Perform the dot product with an input matrix

        :param arr: Matrix to perform dot product.
        :type arr: NDArray
        :return: Matrix result of the dot product.
        :rtype: NDArray
        '''
        
        return self.aff_mat.dot(arr)
        
    @classmethod
    def random(cls, size: int, low: float = 0., high: float = 1.) -> 'AffinityMatrix':
        '''
        Factory method to generate a random affinity matrix of a specified size within a given range.
        The diagonal is one
        
        :param size: Size of the square matrix (number of rows/columns).
        :type size: int
        :param low: Lower bound of the random numbers (inclusive), defaults to 0.
        :type low: float
        :param high: Upper bound of the random numbers (exclusive), defaults to 1.
        :type high: float
        :return: Random affinity matrix.
        '''
        
        # NOTE: In the case the user specifies a negative lower bound
        #       it cannot be compliant with positive pairwise similarities constraint
        
        rand_aff_mat = np.random.uniform(low, high, size=(size, size)) # square matrix
        rand_aff_mat = (rand_aff_mat +  rand_aff_mat.T) / 2            # symmetric
        np.fill_diagonal(rand_aff_mat, 0)                              # 0-diagonal
        
        return AffinityMatrix(aff_mat=rand_aff_mat)
    
    
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
        
        return mean([elements * count for elements, count in self.clusters_counts.items()])
    
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
    
    
class DSClustering:
    '''
    Class for performing DominantSet clustering using
    an affinity matrix.
    
    See: Dominant Sets and Pairwise Clustering', 
         by Massimiliano Pavan and Marcello Pelillo, PAMI 2007.
    '''
        
    def __init__(
        self,
        aff_mat: AffinityMatrix,
        min_elements: int = 1,
        max_iter: int = 1000,
        eps: float = 1e-6,
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
        :param eps: Convergence threshold for the replicator dynamics, defaults to 1e-6.
        :type eps: float, optional
        :param logger: Logger to log clustering progress. If not given MutedLogger is set.
        :type logger: Logger | None, optional
        '''
        
        # Affinity Matrix
        self._aff_mat = aff_mat
        
        # Clustering hyperparameters
        self._min_elements = min_elements
        self._max_iter     = max_iter
        self._eps          = eps
        
        # Default logger
        self._logger = default(logger, MutedLogger())
        
        # Clusters
        self._clusters: DSClusters = DSClusters()
        
    # --- MAGIC METHODS ---
    
    def __str__(self) -> str: 
        return  f'DSClustering[objects: {len(self._aff_mat)}; '\
                f'min_elements: {self._min_elements}; '\
                f'max_iter: {self._max_iter}; '\
                f'eps: {self._eps}]'

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
            x: NDArray | None = None,
            max_iter: int   = 1000,
            eps:  float = 1e-8
        ) -> Tuple[NDArray, np.float32, bool]:
            '''
            Perform one-step replicator dynamics on the given affinity matrix.

            Replicator dynamics is an algorithm used for clustering based on affinity matrix.
            It iteratively updates the distribution of cluster memberships until convergence or
            maximum iteration limit is reached.

            :param a: Affinity matrix representing the pairwise similarities between data points.
            :param a: Initial distribution of cluster memberships. If not provided, a uniform
                      distribution will be used.
            :param max_iter: Maximum number of iterations. Default is 1000.
            :param eps: Convergence threshold. The algorithm stops if the change in distribution
                        between iterations is smaller than this value. Default is 1e-8.
            :return: Tuple containing the final distribution of cluster memberships, their coherence and 
                     a boolean indicating if the process converged.
            '''
            
            # In the case initial distribution is not given use a uniform one
            x = default(x, np.zeros(len(aff_mat)) + 1 / len(aff_mat))

            # Initialize hyperparameters
            dist:  float = 2 * eps  # this is just for entering the first loop
            iter:  int   = 0 

            # Loop until convergence threshold
            # or until predetermined number of iterations            
            while iter < max_iter and dist > eps:
                
                # Store old distribution for convergence comparison
                x_old = np.copy(x)  

                # Promote cluster membership with higher payoffs
                # x = x' A x
                x = x * aff_mat.dot(x)

                # Compute coherence 
                # W_S = \sum_{i \in S} w_S(i)
                w: np.float32 = np.sum(x)
            
                # Normalize the distribution
                x = x / w

                # Compute the change in distribution with previous iterations
                dist = np.sqrt(np.sum((x - x_old) ** 2))

                # Increment the iteration counter
                iter += 1
                
            # Convergence flag
            converged = iter < max_iter
                
            return x, w, converged
    
    def run(self):
        '''
        Run DS clustering algorithm with specified class hyper-parameters
        '''
        
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
            x, w, converged = DSClustering._replicator_dynamics(
                aff_mat=aff_mat, 
                max_iter=self._max_iter
            )
            
            # Warning if it didn't converge
            if not converged:
                wrn_msg = f'Cluster {iter} didn\'t converge in {self._max_iter} iterations. '
                self._logger.warn(wrn_msg)
            
            # Extract the cluster as vector support
            support: NDArray[np.int32] = np.where(x >= self._eps)[0]

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
                
