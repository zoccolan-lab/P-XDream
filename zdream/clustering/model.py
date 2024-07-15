from __future__ import annotations

import math
import os
from typing import Tuple
import numpy as np
from numpy.typing import NDArray, ArrayLike
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from zdream.utils.logger import Logger, SilentLogger
from zdream.utils.misc import default

# --- TYPE ALIAS ---

Label  = np.int32 | np.str_
''' Label associated to an object in an affinity matrix. '''

Labels = NDArray[Label]
''' 
Labels associated to objects in an affinity matrix.
Array length has the same size of matrix side.
'''

class AffinityMatrix:
    '''
    Class representing a 2-dimensional square affinity matrix
    It allows to associate labels to each object and perform
    basic operations such as copying and subsetting
    '''

    def __init__(self, A: NDArray, labels: Labels | None = None) -> None:
        '''
        Instantiate a new object with affinity matrix
        It performs sanity check for shape and values

        :param A: Affinity matrix.
        :type A: NDArray
        :param labels: Labels associated to each object.
                       If not given matrix indexes are used.
        :type labels: Labels
        '''

        # Sanity checks

        # A) Shape check
        if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            err_msg = f'The affinity matrix is supposed to be square, but {A.shape} shape found.'
            raise ValueError(err_msg)

        # B) Symmetric check
        if not np.array_equal(A, A.T):
            err_msg = f'The affinity matrix it\'s not symmetric.'
            raise ValueError(err_msg)

        # C) Zero diagonal check
        if np.any(np.diag(A) != 0):
            err_msg = f'The affinity matrix diagonal contains non-zero values.'
            raise ValueError(err_msg)

        # D) Positive similarities 
        if np.any(A < 0):
            err_msg = f'The affinity matrix contains negative similarities.'
            raise ValueError(err_msg)

        # Save matrix and labels
        self._A      : NDArray = A
        self._labels : Labels  = default(labels, np.array(list(range(len(self)))))

    # --- MAGIC METHODS ---

    def __len__ (self) -> int:  return self.shape[0]
    def __str__ (self) -> str:  return f'AffinityMatrix[objects: {len(self)}]'
    def __repr__(self) -> str:  return str(self)
    def __bool__(self) -> bool: return np.sum(self.A) > 0

    def __copy__(self) -> 'AffinityMatrix':
        return AffinityMatrix(
            A = np.copy(self.A),
            labels  = np.copy(self.labels)
        )

    # --- PROPERTIES ---

    @property
    def A(self)              -> NDArray         : return self._A

    @property
    def labels(self)         -> Labels          : return self._labels

    @property
    def shape(self)          -> Tuple[int, ...] : return self.A.shape


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
        new_mat = np.delete(self._A, ids, axis=0)
        new_mat = np.delete(      new_mat, ids, axis=1)

        # Return new instance if not inplace operation
        if not inplace:
            return AffinityMatrix(A=new_mat, labels=new_labels)

        # Update attributes of current objects
        self._labels  = new_labels
        self._A = new_mat
        
    def save(
        self,
        out_dir: str, 
        file_name: str = 'aff_mat.npy',
        logger: Logger = SilentLogger()
    ):
        '''
        Save affinity matrix to target directory with specified file name.

        :param out_dir: Directory where to store the affinity matrix.
        :type out_dir: str
        :param file_name: Output file name, defaults to 'aff_mat.npy'
        :type file_name: str, optional
        :param logger: Optional logger to log i/o information.
                       If not given muted logger is used.
        :type logger: Logger | None, optional
        '''
        
        # Output directory
        os.makedirs(out_dir, exist_ok=True)
        
        # Save file
        out_fp = os.path.join(out_dir, file_name)
        logger.info(f'Saving affinity matrix to {out_fp}')
        np.save(out_fp, self.A)

    @classmethod
    def from_file(cls, path: str, labels: Labels | None = None) -> 'AffinityMatrix':
        '''
        Load affinity matrix from .NPY file

        :param path: File path to numpy matrix.
        :type path: str
        :return: Loaded affinity matrix
        :rtype: AffinityMatrix
        '''
        
        aff_mat = np.load(file=path)
        
        return AffinityMatrix(A=aff_mat, labels=labels)

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

        return AffinityMatrix(A=rand_aff_mat)


class PairwiseSimilarity:
    '''
    Class for computing pairwise-similarities over a 
    '''
    
    @staticmethod
    def cosine_similarity(matrix: NDArray) -> AffinityMatrix:
        '''
        Compute pairwise similarity with cosine similarity
        '''
        
        aff_mat  = cosine_similarity(matrix) # \in to [-1, 1]
        aff_mat += 1                         # \in to [ 0, 2]
        aff_mat /= 2                         # \in to [ 0, 1]
        
        np.fill_diagonal(aff_mat, 0)        # zero diagonal
        
        return AffinityMatrix(A=aff_mat)
    
