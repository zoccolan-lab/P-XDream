from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Dict, Iterable, List, Set, cast
import numpy as np
from numpy.typing import NDArray

from pxdream.clustering.cluster import Cluster, Clusters
from pxdream.clustering.model import AffinityMatrix, Labels
from pxdream.utils.logger import Logger, SilentLogger
from pxdream.utils.io_ import read_json

from functools import cache


class DS:
    '''
    Class for the computation of DominantSet formulas
    on an affinity matrix.
    '''
    
    def __init__(self, aff_mat: AffinityMatrix) -> None:
        
        self._aff_mat = aff_mat
        
        
    def awdeg(self, S: Set[int], i: int) -> float:
        '''
        Compute the average weighted similarity between node i in S,
        where S is a set of vertexes i belong to.

        :param S: Set of vertexes
        :type S: Set[int]
        :param i: Vertex in S
        :type i: int
        :return: Average weighted similarity between node i and S
        :rtype: float
        '''
        # Convert set to tuple for hashing
        if i not in S:
            raise ValueError(f'Node {i} is not part of S={S}')
        
        return sum([self._aff_mat.A[i, j] for j in S]) / len(S)
    

    def phi(self, S: Set[int], i: int, j: int) -> float:
        '''
        Compute the relative similarity between nodes j and i
        with respect to the average similarity between node i and
        its neighbors in S, with i in S and j not in S

        :param S: Set of vertexes
        :type S: Set[int]
        :param i: Vertex in S
        :type i: int
        :param j: Vertex not in S
        :type j: int
        :return: Relative similarity between nodes j and i w.r.t S
        :rtype: float
        '''
        
        # \phi_S(i, j) = a_{ij} - awdeg_S(i)
        if j in S:
            raise ValueError(f'Node {j} is part of S={S}')
        
        return self._aff_mat.A[i, j] - self.awdeg(S=S, i=i)
    
    def w(self, S: Set[int], i: int) -> float:
        '''
        Compute the coherence of node i in S, 
        where S is a set of vertexes i belong to.
        
        :param S: Set of vertexes
        :type S: Set[int]
        :param i: Vertex in S
        :type i: int
        :return: Internal coherence of i in S
        :rtype: float
        '''
        
        
        if i not in S:
            raise ValueError(f'Node {i} is not part of S={S}')
        
        # w_S(i) = 1, if |S| = 1
        if len(S) == 1:
            return 1.
        
        # w_S(i) = \sum_{j \in S \setminus \{i\}} \phi_{S \setminus \{i\}}}(j, i) w_{S \setminus \{i\}}}(j)
        S_sub = S.difference([i])
        
        return sum([self.phi(S=S_sub, i=j, j=i) * self.w(S=S_sub, i=j) for j in S_sub])
    
    @cache
    def W(self, S: Set[int]) -> float:
        '''
        Compute the total coherence of set of nodes S 
        
        :param S: Set of vertexes
        :type S: Set[int]
        :return: Total weight of s
        :rtype: float
        '''
        
        # W(S) = \sum_{i \in S} w_S(i)
        return sum(self.w(S=S, i=i) for i in S)



class DSCluster(Cluster):
    ''' 
    Class representing a cluster result of Dominant Set clustering
    It collects a list of object identified by a label and their rank inside the group
    and a scalar representing the total internal coherence of the group.
    '''
    
    @dataclass
    class DSClusterObject(Cluster.ClusterObject):
        '''
        Class representing an object inside the DS cluster
        with two attributes: 
        - the label identifying the object (inherits from ClusterObject).
        - the rank inside the cluster.
        
        Object in the cluster are sorted by decreasing rank.
        '''
        
        rank  : np.float32
        ''' Rank of the object inside the cluster. Normally in [0, 1] '''
        
        def __str__ (self) -> str: return f'{super().__str__()[:-1]}; rank: {self.rank}]'
        
        @property
        def info(self) -> Dict[str, Any]: 
            ''' Add rank to object information '''
            
            super_info = super().info
            super_info['rank'] = self.rank.tolist()  # tolist() is used to convert numpy.float32 to float
            
            return super_info
        
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
        
        # Objects with decreasing Rank
        self._objects = [self.DSClusterObject(label=label, rank=rank) for label, rank in zip(labels, ranks)]
        self._objects.sort(key=lambda obj: obj.rank, reverse=True)
    
        self._w = w
    
    # --- MAGIC METHODS ---
    
    def __str__(self) -> str : return f'DS{super().__str__()[:-1]}; W: {self.W}]'
    
    # NOTE: We need to cast the return type of the following methods to subclass
    
    def __iter__   (self)           -> Iterable[DSClusterObject] : return cast(Iterable[DSCluster.DSClusterObject], super().__iter__())
    def __getitem__(self, idx: int) -> DSClusterObject           : return cast(DSCluster.DSClusterObject,           super().__getitem__(idx=idx))
    
    # --- PROPERTIES ---
    
    # NOTE: We need to cast the return type method to subclass
    @property
    def objects(self) -> List[DSClusterObject]: return cast(List[DSCluster.DSClusterObject], super().objects)
    ''' List of objects in the cluster''' 
    
    @property
    def W(self) -> float: return float(self._w)
    ''' Coherence of the cluster'''

    @property
    def ranks(self) -> NDArray[np.float32]: return np.array([obj.rank  for obj in self]) # type: ignore
    ''' Ranks of the objects in the cluster'''
        
    
    @property
    def info(self) -> Dict[str, List[Dict[str, Any]] | Any]:
        ''' Add coherence to cluster information '''
        
        super_info = super().info
        super_info['W'] = self.W
        
        return super_info
    
    # --- UTILS ---

    def units_mapping(self, arr: NDArray, weighted: bool = True) -> NDArray:
        '''
        Return the units activation mapping for clusters units.

        :param arr: Array with units to map
        :type arr: NDArray
        :param weighted: If to use cluster weights, defaults to True
            If not set, an uniform distribution is used.
        :type weighted: bool, optional
        :return: _description_
        :rtype: UnitsMapping
        '''
        
        if weighted : weights = self.ranks                     # Weighted mean
        else        : weights = np.ones(len(self)) / len(self) # Arithmetic mean

        # Remap units
        arr[:, self.labels] *= weights # type: ignore

        return arr
    
    @staticmethod
    def extract_singletons(aff_mat: AffinityMatrix) -> List[DSCluster]:
        '''
        Extract trivial clusters with one element from an Affinity matrix.
        Their score is 1 and the coherence is 0. 

        :return: List of singleton clusters clusters 
        :rtype: List[DSCluster]
        '''
        
        # NOTE: The case case correspond to an AffinityMatrix of just
        #       one element. Since the only value which is also the 
        #       diagonal is equal to 0, then also the coherence w must be so.
        return [
            DSCluster(
                labels = np.array([lbl]), 
                ranks  = np.array([1.]),
                w      = np.float32(0.)
            )
            for lbl in aff_mat.labels
        ]


class DSClusters(Clusters):
    '''
    Class for manipulating a set of `DSCluster` objects
    extracted from the same learning algorithm.
    
    The class compute some statistics on clusters and 
    allow to dump them to file.
    '''
    
    NAME = 'DSClusters'
    
    def __init__(
        self, 
        clusters: List[DSCluster] | None = None
    )-> None:
        if clusters is None: clusters = []
        super().__init__(clusters=clusters)  # type: ignore
    
    
    # --- MAGIC METHODS ---   
    
    def __str__    (self)           -> str                 : return f'DS{super().__str__()}'
    def __iter__   (self)           -> Iterable[DSCluster] : return cast(Iterable[DSCluster], super().__iter__()) 
    def __getitem__(self, idx: int) -> DSCluster           : return cast(DSCluster,           super().__getitem__(idx=idx))
    
    # --- PROPERTIES ---
    
    @property
    def clusters(self) -> List[DSCluster]: return cast(List[DSCluster], self._clusters)
    
    # --- UTILITIES ---
    
    def add(self, cluster: DSCluster): super().add(cluster=cluster)

    
    @classmethod
    def from_file(cls, fp: str, logger: Logger = SilentLogger()) -> DSClusters:
        '''
        Load a DSCluster state from JSON file
        
        :param fp: File path to .JSON file where DSClusters information is stored.
        :type fp: str
        :param logger: Logger to log i/o information.
        :type logger: Logger
        :return: Loaded clusters object.
        :rtype: DSClusters
        '''
        
        logger.info(msg=f'Reading clusters from {fp}')
        data = read_json(path=fp)
        
        clusters = [
            DSCluster(
                labels = np.array([obj['label'] for obj in cluster['objects']], dtype=np.int32),
                ranks  = np.array([obj['rank' ] for obj in cluster['objects']]),
                w      = np.float32(cluster['W'])
            )
            for cluster in data.values()     
        ]
            
        return DSClusters(clusters=clusters)
