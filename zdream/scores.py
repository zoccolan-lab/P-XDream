from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Literal, cast
from functools import partial

from einops import reduce
import numpy as np
from numpy.typing import NDArray

from .utils import SubjectScore, default, is_multiple_state
from .utils import ObjectiveFunction, SubjectState


def distance(
    ord : float | Literal['fro', 'nuc'] |  None = 2,
    axis : int | None = -1,
) -> ObjectiveFunction:

    axis = cast(int, axis)
    norm = partial(np.linalg.norm, axis=axis, ord=ord)
    
    return cast(ObjectiveFunction, norm)

def mse(arr1: NDArray, arr2: NDArray) -> NDArray[np.float32]:
        '''
        Compute the Mean Squared Error for the two batches of image in input

        :param arr1: first array with shape [B, C, W, H]
        :type arr1: NDArray
        :param arr2: second array with shape [B, C, W, H]
        :type arr2: NDArray
        :return: _description_
        :rtype: NDArray[np.float32]
        '''
        
        # TODO Explicit check for dimension
        
        return np.mean(np.square(arr1 - arr2), axis=(1, 2, 3)).astype(np.float32)


class Score(ABC):
    '''
    '''

    def __init__(
        self,
        criterion : ObjectiveFunction
    ) -> None:
        self.criterion = criterion

    @abstractmethod
    def __call__(self, state : SubjectState) -> SubjectScore:
        pass
    
class MSEScore(Score):
    '''
        Class simulating a neuron score which target state 
        across one or multiple layers can be set.
        The scoring function is the MSE with the target.
    '''
    
    def __init__(
        self,
        target : SubjectState
    ) -> None:
        '''
        The constructor uses the target state to create
        the objective function.

        :param target: target state or dictionary of target states 
                        indexed by layer name
        :type target: SubjectState
        '''
        
        self._target : SubjectState = target
        
        mse_image = partial(self._images_mse, state2=self._target)
        
        super().__init__(criterion=mse_image)
        
    def __call__(self, state : SubjectState) -> SubjectScore:
        '''
        Compute the score function with a given subject state in terms of MSE

        :param state: state of subject state as one array or multiple
                        arrays indexed by layer name
        :type state: SubjectState
        :return: array of scores
        :rtype: SubjectScore
        '''
        
        score = self.criterion(state)
        
        print(type(state))
        
        # NOTE the minus sign is required because the score is passed
        #      to an optimizer which is a maximizer; the MSE returns non 
        #      negative values where 0 corresponds to the perfect match       
        if is_multiple_state(state=state):
            score = {k: -v for k, v in cast(Dict[str, NDArray], score).items()}
        else:
            score = - cast(NDArray, score)
        
        return score
    
    
    @staticmethod
    def _images_mse(state1: SubjectState, state2: SubjectState) -> SubjectScore:
        
        
        # Array state
        if  not is_multiple_state(state1) and\
            not is_multiple_state(state2):
            
            state1 = cast(NDArray, state1)
            state2 = cast(NDArray, state2)

            score = mse(arr1=state1, arr2=state2)
            
            return score
        
        # Dict state
        if  is_multiple_state(state1) and\
            is_multiple_state(state2):
            
            state1 = cast(Dict[str, NDArray], state1)
            state2 = cast(Dict[str, NDArray], state2)
            # Check for layer name consistency
            # NOTE it can raise an error if keys are not consistent
            #      the state1 is expected to be the test one
            # TODO make explicit check for keys consistency
            scores = {
                layer: mse(arr1=state1[layer], arr2=state2[layer]) for layer in state1.keys()
            }
            
            return scores
        
        # Incompatible states type
        err_msg = 'The states are expected both to be both arrays or both dicts'
        raise ValueError(err_msg)
    
    
class MinMaxSimilarity(Score):
    '''
    '''

    def __init__(
        self,
        positive_target : str,
        negative_target : str,
        neg_penalty : float = 1.,
        similarity_fn : ObjectiveFunction | None = None,
        grouping_fn : Callable[[NDArray], Tuple[NDArray, NDArray]] | None = None,
    ) -> None: 
        
        # If similarity function is not given use euclidean distance as default
        euclidean = distance(ord=2)
        criterion = default(similarity_fn, euclidean)
        
        # If grouping function is not given use even-odd split as default
        group_fun = default(grouping_fn, lambda x : (x[::2], x[1::2]))

        super().__init__(criterion=criterion)

        self.positive_target = positive_target
        self.negative_target = negative_target

        self.grouping_fn = group_fun
        self.neg_penalty = neg_penalty

    def __call__(self, state: SubjectState) -> NDArray[np.float32]:
        if not isinstance(state, dict):
            err_msg = 'MinMaxSimilarity expects subject state to be a dictionary'
            raise ValueError(err_msg)
        
        pos_a, pos_b = self.grouping_fn(state[self.positive_target])
        neg_a, neg_b = self.grouping_fn(state[self.negative_target])

        score = self.criterion(pos_a - pos_b) - self.neg_penalty * self.criterion(neg_a - neg_b)

        return score


