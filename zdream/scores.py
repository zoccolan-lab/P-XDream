import numpy as np
from abc import abstractmethod

from .utils import default
from .utils import ObjectiveFunction, SubjectState

from einops import reduce
from functools import partial

from typing import Callable, Tuple, Literal, cast
from numpy.typing import NDArray

def distance(
    ord : float | Literal['fro', 'nuc'] |  None = 2,
    axis : int | None = -1,
) -> ObjectiveFunction:
    axis = cast(int, axis)
    norm = partial(np.linalg.norm, axis=axis, ord=ord)
    
    return cast(ObjectiveFunction, norm)

class Score:
    '''
    '''

    def __init__(
        self,
        criterion : ObjectiveFunction
    ) -> None:
        self.criterion = criterion

    @abstractmethod
    def __call__(self, state : SubjectState) -> NDArray[np.float32]:
        raise NotImplementedError('Score __call__ method is abstract')
    
class MinMaxSimilarity(Score):
    '''
    '''

    def __init__(
        self,
        positive_target : str,
        negative_target : str,
        neg_penalty : float = 1.,
        similarity_fn : ObjectiveFunction | None = None,
        grouping_fn : Callable[[NDArray], Tuple[NDArray, ...]] | None = None,
    ) -> None:
        euclidean = distance(ord=2)
        criterion = default(similarity_fn, euclidean)
        group_fun = default(grouping_fn, lambda x : (x[::2], x[1::2]))

        super().__init__(criterion)

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
