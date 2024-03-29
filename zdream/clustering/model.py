from __future__ import annotations

import os
from typing import Any, Callable, List, Tuple
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity

from zdream.logger import Logger, MutedLogger
from zdream.subject import InSilicoSubject
from zdream.utils.misc import default, device
from zdream.message import ZdreamMessage

# --- TYPE ALIAS ---

Label  = np.int32 | np.str_
''' Label associated to an object in an affinity matrix. '''

Labels = NDArray[Label]
''' 
Labels associated to objects in an affinity matrix.
Array length has the same size of matrix side.
'''

Recording = NDArray[np.float32]
'''
Two dimensional array of dimension NEURONS x STIMULI
The element (i, j) indicates that the neuron i produced
the specific activation when presented stimulus j 
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
        self._A: NDArray = A
        self._labels : Labels  = default(labels, np.array(list(range(len(self)))))

    # --- MAGIC METHODS ---

    def __len__ (self) -> int: return self.shape[0]
    def __str__ (self) -> str: return f'AffinityMatrix[objects: {len(self)}]'
    def __repr__(self) -> str: return str(self)

    def __copy__(self) -> 'AffinityMatrix':
        return AffinityMatrix(
            A = np.copy(self.A),
            labels  = np.copy(self.labels)
        )

    # --- PROPERTIES ---

    @property
    def A(self) -> NDArray: return self._A

    @property
    def labels(self) -> Labels: return self._labels

    @property
    def shape(self) -> Tuple[int, ...]: return self.A.shape
    
    @property
    def max_eigenvalue(self) -> float: return float(np.max(np.linalg.eigvals(self.A))
)
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
        logger: Logger = MutedLogger()
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
    

class NeuronalRecording:
    '''
    Class for neuronal recording from an image dataset
    '''
    
    def __init__(
        self, 
        subject:       InSilicoSubject, 
        dataset:       Dataset, 
        image_ids:     List[int]               = [], 
        stimulus_post: Callable[[Any], Tensor] = (lambda x: x),
        state_post:    Callable[[Any], Tensor] = (lambda x: x),
        logger:        Logger                  = MutedLogger()
    ):
        '''
        Instantiate a new recording with a given subject and 

        :param subject: Subject to present stimuli.
        :type subject: InSilicoSubject
        :param dataset: Dataset of visual stimuli
        :type dataset: Dataset
        :param image_ids: Indexes of stimuli to present.
                          If not specified the entire dataset is used.
        :type image_ids: List[int] | None
        :param stimulus_post: Postprocessing for stimulus, defaults to identity.
        :type stimulus_post: Callable[[Any], Tensor] | None
        :param state_post: Postprocessing for subject state, defaults to identity.
        :type state_post: Callable[[Any], Tensor] | None
        :param logger: Logger to log processing information, defaults to MutedLogger.
        :type logger: Logger | None, optional
        '''
        
        # Input parameters
        self._subject       = subject
        self._dataset       = dataset
        self._stimulus_post = stimulus_post
        self._state_post    = state_post
        self._logger        = logger
        self._image_ids     = image_ids if image_ids else list(range(len(dataset))) # type: ignore
        
        self._recording: Recording = np.array([], dtype=np.float32)
        
    def __str__(self) -> str:
        return f'NeuralRecording[subject: {self._subject}; n-stimuli: {len(self._image_ids)}]'\
        
    @property
    def recordings(self) -> Recording:
        if self._recording.size:
            return self._recording
        raise ValueError(f'Recordings were not computed yet')
    
    def record(self, log_chk: int | None = None):
        '''
        Perform subject recording by presenting visual 
        stimuli with specified index.

        :param log_chk: Iteration for logging progress.
        :type log_chk: int | None
        '''
        
        # If not given we set checkpoint progress greater that iterations
        log_chk = default(log_chk, len(self._image_ids)+1)
        
        sbj_states = []
        msg = ZdreamMessage()
        
        for i, idx in enumerate(self._image_ids):
            
            # Log progress
            if i % log_chk == 0:
                progress = f'{i:>{len(str(len(self._image_ids)))+1}}/{len(self._image_ids)}'
                perc     = f'{i * 100 / len(self._image_ids):>5.2f}%'
                self._logger.info(mess=f'Iteration [{progress}] ({perc})')
            
            # Retrieve stimulus
            stimulus: Tensor = self._dataset[idx]
            stimulus = self._stimulus_post(stimulus).to(device)
            
            # Compute subject state
            try: 
                sbj_state, _ = self._subject(data=(stimulus, msg))
                sbj_state = self._state_post(sbj_state)
            except Exception as e:
                self._logger.warn(f"Unable to process image with index {idx}: {e}")
                
            # Update states
            sbj_states.append(sbj_state)
        
        # Save states in recordings
        self._recording = np.stack(sbj_states).T
        
    def save(self, out_dir: str, file_name: str = 'recordings.npy'):
        '''
        Save affinity matrix to target directory with specified file name.

        :param out_dir: Directory where to store the recordings matrix.
        :type out_dir: str
        :param file_name: Output file name, defaults to 'recordings.npy'
        :type file_name: str, optional
        :param logger: Optional logger to log i/o information.
                       If not given muted logger is used.
        :type logger: Logger | None, optional
        '''
        
        os.makedirs(out_dir, exist_ok=True)
        
        out_fp = os.path.join(out_dir, file_name)
        
        self._logger.info(f'Saving recordings to {out_fp}')
        
        np.save(out_fp, self.recordings)
        

class PairwiseSimilarity:
    '''
    Class for computing pairwise-similarities over a recordings.
    '''
    
    def __init__(self, recordings: Recording) -> None:
        '''
        Create a new instance of PairwiseSimilarity to compute over neural recordings.

        :param recordings: Matrix of neuronal recordings
        :type recordings: Recording
        :param logger: lo, defaults to None
        :type logger: Logger | None, optional
        '''
        
        self._recordings = recordings
        
    def __len__(self) -> int:
        return self._recordings.shape[0]
    
    @property
    def cosine_similarity(self) -> AffinityMatrix:
        '''
        Compute pairwise similarity with cosine similarity
        '''
        
        aff_mat = (cosine_similarity(self._recordings) + np.ones(len(self))) / 2
        np.fill_diagonal(aff_mat, 0)
        
        return AffinityMatrix(A=aff_mat) 