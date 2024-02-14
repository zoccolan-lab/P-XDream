from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Literal, cast
from functools import partial

from einops import reduce
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.functional import mse_loss as mse

from .utils import default
from .utils import ObjectiveFunction, SubjectState


def distance(
    ord : float | Literal['fro', 'nuc'] |  None = 2,
    axis : int | None = -1,
) -> ObjectiveFunction:

    axis = cast(int, axis)
    norm = partial(np.linalg.norm, axis=axis, ord=ord)
    
    return cast(ObjectiveFunction, norm)

def mse_torch_numpy(
    input  : Tensor,
    target : Tensor 
) -> NDArray:
    
    return mse(input=input, target=target).numpy()

class Score(ABC):
    '''
    '''

    def __init__(
        self,
        criterion : ObjectiveFunction
    ) -> None:
        self.criterion = criterion

    @abstractmethod
    def __call__(self, state : SubjectState) -> NDArray[np.float32]:
        pass
    
class ImageScore(Score):
    '''
        Class simulating a neuron which preferred
        stimulus is a known image. The score function
        is the pixel MSE
    '''
    
    def __init__(
        self,
        image : Tensor
    ) -> None:
        '''
        The init function defines 

        :param image: preferred target stimulus
        :type image: Tensor
        '''
        
        mse_image = partial(mse_torch_numpy, target=image)
        
        super().__init__(criterion=cast(ObjectiveFunction, mse_image))
        
    def __call__(self, state : SubjectState) -> NDArray[np.float32]:
        
        if isinstance(state, dict):
            
            return np.concatenate(
                [self.criterion(s) for s in state.values()]
            )
        
        return self.criterion(state)
        
    
    
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


