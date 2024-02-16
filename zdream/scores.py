from abc import ABC
from typing import Callable, Dict, List, Tuple, cast
from functools import partial
from itertools import combinations

from einops import reduce
import numpy as np
from numpy.typing import NDArray

from scipy.spatial.distance import euclidean

from .utils import default
from .utils import Message
from .utils import SubjectScore, SubjectState

ScoringFunction   = Callable[[SubjectState], Dict[str, SubjectScore]]
AggregateFunction = Callable[[Dict[str, SubjectScore]], SubjectScore]

def distance(
    UV : Tuple[NDArray, NDArray],
    metric : str = 'euclidean',
) -> SubjectScore:

    match metric:
        case 'euclidean': dist = euclidean
        case _: raise ValueError(f'Unrecognized metric {metric}')

    return np.array([dist(u, v) for u, v in zip(*UV)]) 

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
        
        # TODO Explicit check for dimension or use NDArray typing
        
        return np.mean(np.square(arr1 - arr2), axis=(1, 2, 3)).astype(np.float32)


class Score(ABC):
    '''
    Abstract class for computing subject scores.
    '''

    def __init__(
        self,
        criterion : ScoringFunction,
        aggregate : AggregateFunction,
    ) -> None:
        self.criterion = criterion
        self.aggregate = aggregate

    def __call__(self, data : Tuple[SubjectState, Message]) -> Tuple[SubjectScore, Message]:
        '''
        Compute the subject scores given a subject state.

        :param state: state of subject
        :type state: SubjectState
        :return: array of scores
        :rtype: SubjectScore
        '''
        state, msg = data
        
        scores = self.criterion(state)
        
        return (self.aggregate(scores), msg)
    
class MSEScore(Score):
    '''
        Class simulating a neuron score which target state 
        across one or multiple layers can be set.
        The scoring function is the MSE with the target.
    '''
    
    def __init__(
        self,
        target : SubjectState,
        aggregate : AggregateFunction | None = None,
    ) -> None:
        '''
        The constructor uses the target state to create
        the objective function.

        :param target: target state or dictionary of target states 
                        indexed by layer name
        :type target: SubjectState
        '''
        
        self._target : SubjectState = target
        
        # MSEScore criterion is the mse between the measured subject
        # state and a given fixed target. This is accomplished via the
        # partial higher order function that fixes the second input to
        # the _score method of the class
        criterion = partial(self._score, target=self._target)
        aggregate = default(aggregate, lambda d : cast(NDArray, np.mean(list(d.values()), axis=0)))
        
        super().__init__(
            criterion=criterion,
            aggregate=aggregate,    
        )
        
    def _score(self, state: SubjectState, target: SubjectState) -> Dict[str, SubjectScore]:
        # Check for layer name consistency
        if not set(state.keys()).issubset(set(target.keys())):
            raise AssertionError('Keys of test image not in target')
        
        scores = {
            layer: -mse(arr1=state[layer], arr2=target[layer]) for layer in state.keys()
        }
                
        return scores
    
class NeuronScore(Score):
    
    def __init__(
            self, 
            neurons: Dict[str, List[Tuple[int, int]]], 
            aggregate: AggregateFunction
        ) -> None:
        
        criterion = partial(self._aux, neurons=neurons)
                
        super().__init__(criterion, aggregate)
        
    def _aux(self, state: SubjectState, neurons: Dict[str, List[Tuple[int, int]]]) -> Dict[str, SubjectScore]:
        
        # Check for layer name consistency
        if not set(state.keys()).issubset(set(neurons.keys())):
            err_msg = f'Keys of test image not in target {set(state.keys()).difference(neurons.keys())}'
            raise AssertionError(err_msg)
        
        scores = {
            layer: np.sum([activations[i][j] for i, j in neurons[layer]]) for layer, activations in state.keys()
        }
        
        return scores    
    
    
class WeightedPairSimilarity(Score):
    '''
    This scorer computes weighted similarities (negative distances)
    between groups of subject states. Weights can either be positive
    or negative. Groups are defined via a grouping function. 
    '''

    # TODO: Change pairing function mechanism
    def __init__(
        self,
        signature : Dict[str, float],
        metric : str = 'euclidean',
        pair_fn : Callable[[NDArray], Tuple[NDArray, NDArray]] | None = None,
    ) -> None: 
        '''
        
        :param signature: Dictionary containing for each recorded state
            (the str key in the dict) the corresponding weight (a float)
            to be used in the final aggregation step. Positive weights
            (> 0) denote desired similarity, while negative weights (< 0)
            denote desired dissimilarity.
        :type signature: Dictionary of string with float values
        :param metric: Which kind of distance to use. Should be one of
            the supported scipy.spatial.distance function
        :type metric: string
        :param pair_fn: Grouping function defining (within a given state,
            i.e. a given recorded layer) which set of activations should be
            compared against.
            Default: Odd-even pairing, i.e. given the subject state:
                state = {
                    'layer_1' : [+0, +1, -1, +3, +2, +4],
                    'layer_2' : [-4, -5, +9, -2, -7, +8],
                }
            the default pairing build the following pairs:
                'layer_1' : {
                    'l1_pair_1: [+0, -1, +2],
                    'l1_pair_2: [+1, +3, +4],
                }
                'layer_2' : {
                    'l2_pair_1': [-4, +9, -7],
                    'l2_pair_2': [-5, -2, +8],
                }
            so that the similarities will be computed as:
                'layer_1' : similarity(l1_pair_1, l1_pair_2)
                'layer_2' : similarity(l2_pair_1, l2_pair_2)
        :type pair_fn: Callable[[NDArray], Tuple[NDArray, NDArray]] | None
        '''
        
        # If grouping function is not given use even-odd split as default
        # pair_fn = default(pair_fn, lambda x : (x[::2], x[1::2]))
        pair_fn = default(
            pair_fn, 
            lambda x: (
                np.stack([i for i, _ in combinations(x, 2)]), 
                np.stack([j for _, j in combinations(x, 2)])
            )
        )
                
        

        # If similarity function is not given use euclidean distance as default
        self._metric = partial(distance, metric=metric)
        
        criterion = partial(self._score, pair_fn=pair_fn)
        aggregate = partial(self._dprod, signature=signature)
        
        self.signature = signature

        super().__init__(
            criterion=criterion,
            aggregate=aggregate,
        )

    def _score(
        self,
        state : SubjectState,
        pair_fn : Callable[[NDArray], Tuple[NDArray, NDArray]]
    ) -> Dict[str, NDArray]:
        scores = {
            k: -self._metric(pair_fn(v)) for k, v in state.items()
        }
        
        return scores

    def _dprod(self, state : Dict[str, SubjectScore], signature : Dict[str, float]) -> SubjectScore:
        return cast(
            SubjectScore,
            np.sum(
                [v * state[k] for k, v in signature.items()],
                axis=0
            )
        )
        

