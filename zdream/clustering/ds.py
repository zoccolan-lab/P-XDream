from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from os import path
from statistics import mean
from typing import Any, Counter, Dict, Iterable, List, Set, Tuple, cast
import numpy as np
from numpy.typing import NDArray

from zdream.clustering.model import AffinityMatrix, Label, Labels
from zdream.logger import Logger, MutedLogger
from zdream.utils.io_ import read_json, save_json
from zdream.utils.misc import default

from functools import cache

from zdream.utils.model import LayerReduction, Score, ScoringUnit, State, UnitsMapping

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
        
        def __str__ (self) -> str: return f'DSObject[label: {self.label}; rank: {self.rank})]'
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
    
    def __str__ (self) -> str: return f'DSCluster[objects: {len(self)}, coherence: {self.W}]'
    def __repr__(self) -> str: return str(self)
    def __len__ (self) -> int: return len(self.objects)
    
    def __iter__(self) -> Iterable[DSObject]:    return iter(self.objects)
    def __getitem__(self, idx: int) -> DSObject: return self.objects[idx]
    
    # --- PROPERTIES ---
    
    @property
    def W(self) -> float: return float(self._w)
    
    @property
    def objects(self) -> List[DSObject]: return self._objects

    @property
    def labels(self) -> Labels: 
        return np.array([obj.label for obj in self]) # type: ignore

    @property
    def ranks(self) -> NDArray[np.float32]: 
        return np.array([obj.rank  for obj in self]) # type: ignore

    @property
    def scoring_units(self) -> ScoringUnit: return list(self.labels) 
    
    # --- UTILS ---

    def units_mapping(self, arr: NDArray, weighted: bool = True) -> NDArray:
        '''
        Return the units activation mapping for clusters units.

        :param arr: Array with units to map
        :type arr: NDArray
        :param weighted: If to use cluster weights, defaults to True
                         If no uses, an uniform distribution is used.
        :type weighted: bool, optional
        :return: _description_
        :rtype: UnitsMapping
        '''
        
        # Weighted mean
        if weighted:
            weights = self.ranks
        
        # Arithmetic mean
        else:
            weights = np.ones(len(self)) / len(self)

        # Remap units
        arr[:, self.labels] *= weights # type: ignore

        return arr
    
    @staticmethod
    def extract_singletons(aff_mat: AffinityMatrix) -> List[DSCluster]:
        '''
        Extract trivial clusters with one element from an Affinity matrix.
        Their score and coherence is 1. 

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
    
    def __str__ (self) -> str:                 return f'DSClusters[n-clusters: {len(self)}]'
    def __repr__(self) -> str:                 return str(self)
    def __len__ (self) -> int:                 return len (self.clusters)
    def __iter__(self) -> Iterable[DSCluster]: return iter(self.clusters)
    
    def __getitem__(self, idx: int) -> DSCluster: return self.clusters[idx]
    
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
    def obj_tot_count(self) -> int:
        '''
        Return the total number of element in the cluster collection

        :return: Cluster cardinality mapping.
        :rtype: Dict[int, int]
        '''
        
        return sum([element * count for element, count in self.clusters_counts.items()])
    
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
                "W": cluster.W
            }
            for i, cluster in enumerate(self) # type: ignore
        } 
        
        fp = path.join(out_fp, 'DSClusters.json')
        logger.info(f'Saving clusters info to {fp}')
        save_json(data=out_dict, path=fp)
        
    @classmethod
    def from_file(cls, fp: str, logger: Logger = MutedLogger()) -> DSClusters:
        '''
        Load a DSCluster state from JSON file

        :param fp: File path to .JSON file where DSClusters information is stored.
        :type fp: str
        :param logger: Logger to log i/o information.
        :type logger: Logger
        :return: Loaded clusters object.
        :rtype: DSClusters
        '''
        
        logger.info(mess=f'Reding clusters from {fp}')
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