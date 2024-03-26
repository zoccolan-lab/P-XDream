from _collections_abc import dict_keys
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Tuple, cast, Literal

import numpy as np
from einops import rearrange
from einops import reduce
from einops.einops import Reduction
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from zdream.utils.model import UnitsReduction, UnitsMapping

from .utils.model import LayerReduction, ScoringUnit, UnitsMapping
from .utils.model import Score
from .utils.model import State
from .utils.misc import default
from .message import Message

# NOTE: This is the same type of _MetricKind from scipy.spatial.distance
#       which we need to redefine for issues with importing private variables from modules.
_MetricKind = Literal[
    'braycurtis', 'canberra', 'chebychev', 'chebyshev',
    'cheby', 'cheb', 'ch', 'cityblock', 'cblock', 'cb',
    'c', 'correlation', 'co', 'cosine', 'cos', 'dice',
    'euclidean', 'euclid', 'eu', 'e', 'hamming', 'hamm',
    'ha', 'h', 'minkowski', 'mi', 'm', 'pnorm', 'jaccard',
    'jacc', 'ja', 'j', 'jensenshannon', 'js', 'kulczynski1',
    'mahalanobis', 'mahal', 'mah', 'rogerstanimoto', 'russellrao',
    'seuclidean', 'se', 's', 'sokalmichener', 'sokalsneath',
    'sqeuclidean', 'sqe', 'sqeuclid', 'yule'
]

class Scorer(ABC):
    '''
    Abstract class for computing stimuli scores given subject states. 

    The score is break in three steps:

    1. Mapping activations - a function for activation element-wise mapping for each layer.
    2. Reducing unit - a function that reduces multiple units of a single layer into
                       a single score for that specific layer.
    3. Reducing layer - a function that reduces scores for different layer into a single
                        score associated to a particular stimulus.
    '''
    
    # Name used in string object representation
    _NAME = 'Scorer'


    def __init__(
        self,
        units_reduce : UnitsReduction,
        layer_reduce : LayerReduction,
        units_map    : UnitsMapping = lambda x: x,
    ) -> None:
        '''
        Initialize a scoring by defining its three basic operations.

        :param units_reduce: Function implementing the logic for computing
                             the score of a single layer across different units.
        :param layer_reduction: Reducing function across layer scores.
        :param units_map: Mapping function across activation units, defaults to identity.
        '''

        self._units_map    : UnitsMapping   = units_map
        self._layer_reduce : LayerReduction  = layer_reduce
        self._units_reduce : UnitsReduction = units_reduce


    def __call__(self, data : Tuple[State, Message]) -> Tuple[Score, Message]:
        '''
        Compute the subject scores given a subject state by 
        using the mapping and reducing functions.

        :param data: The state of the subject and the message.
        :type state: State
        :return: Array of stimuli scores and the message.
        :rtype: Score
        '''
        
        state, msg = data

        # 1. Mapping activations
        state_mapped: State  = {
            layer: self._units_map(act) 
            for layer, act in state.items()
        }
        
        # 2. Performing units reduction
        layer_scores: Dict[str, Score] = self._units_reduce(state_mapped)
        
        # 3. Performing layer reduction
        scores = self._layer_reduce(layer_scores)
        
        return (scores, msg)
    
    # --- MAGIC METHODS ---
        
    def __str__(self) -> str:
        
        dims = ", ".join([f"{k}: {len(v)} units" for k, v in self.scoring_units.items()])
        return f'{self._NAME}[target size: ({dims})]'
    
    def __repr__(self) -> str: return str(self)

    # --- PROPERTIES ---

    @property
    @abstractmethod
    def scoring_units(self) -> Dict[str, ScoringUnit]:
        '''
        Returns the scoring units associated to each layer
        The units are referred to index in the activation

        :return: Scoring units across layers.
        :rtype: Dict[str, ScoringUnit]
        '''
        pass

    @property
    def n_scoring_units(self) -> int:
        ''' 
        Returns the total number of scoring units

        :return: 
        :rtype:
        '''

        return sum(
            [len(v) for v in self.scoring_units.values()]
        )

    # --- UTILITIES ---    

    @staticmethod
    def _dict_values_reduction(data: Dict[str, NDArray], reduction: Reduction) -> NDArray:
        '''
        Utility function to apply reduction over values of a dictionary
        '''

        return reduce(
            np.stack(list(data.values())),    # concatenate values
            pattern='l b -> b',               # perform reduction on the first dimension
            reduction=reduction,
        )   
        
    def _check_key_consistency(self, scoring: dict_keys, state: dict_keys):
        '''
        Check subject state to contains at least the target keys for computing the scores of interest. 
        Additional keys in subject (i.e. layers only for recording) will be ignored by the scorer.

        :param scoring: Scoring layers
        :type scoring: dict_keys
        :param state: State layers
        :type state: dict_keys
        :raises ValueError: If scoring layers were not recorded.
        '''
        
        if not set(scoring).issubset(set(state)):
            err_msg = f'Keys of test image not in target {set(state).difference(scoring)}'
            raise ValueError(err_msg)

class MSEScorer(Scorer):
    '''
    Class simulating a unit which target stimuli is set.
    The mapping function is the MSE with the target stimuli.
    '''
    
    name = 'MSEScorer'
    
    def __init__(
        self,
        target          : State,
        layer_reduction : Reduction = 'mean'
    ) -> None:
        '''
        The constructor only requires the target and the scoring function.
        The mapping function is the MSE form the target.

        :param target: Target state for the neuron.
        :type target: State
        :param layer_reduction: Reducing function across layer scores.
        :type layer_reduction: ScoringReducing
        '''
        
        self._target : State = target
        
        # The mapping function is the MSE between the subject state
        # and the fixed target. 
        units_reduce: UnitsReduction = partial(self._mse_map, target=self._target)

        layer_reduce: LayerReduction = partial(
            self._dict_values_reduction,
            reduction=layer_reduction
        )
        
        super().__init__(
            units_reduce=units_reduce,
            layer_reduce=layer_reduce
        )
        
    def _mse_map(self, state: State, target: State) -> Dict[str, Score]:
        '''
        Compute MSE across layer between state and target.

        :param state: Recorded subject state.
        :type state: State
        :param target: Fixed target.
        :type target: State
        :return: MSE between target and state.
        :rtype: Dict[str, Score]
        '''

        # Check for layer name consistency
        self._check_key_consistency(scoring=target.keys(), state=state.keys())
        
        scores = {
            layer: - self.mse(state[layer], target[layer]) for layer in state.keys()
            if layer in target
        }
                
        return scores
    
    @staticmethod
    def mse(a : NDArray, b : NDArray) -> NDArray:
        ''' Static function to compute MSE between two vectors'''

        # Arrays flattening
        a = rearrange(a, 'b ... -> b (...)')
        b = rearrange(b, 'b ... -> b (...)')

        return np.mean(np.square(a - b), axis=1).astype(np.float32)
    
    @property
    def target(self) -> State:
        return self._target
    
    @property
    def scoring_units(self) -> Dict[str, ScoringUnit]:

        return {k: list(range(np.prod(v.shape))) for k, v in self._target.items()}

    
class ActivityScorer(Scorer):
    '''
    Scorer class to compute a single cross 
    aggregating across multiple units in different layers
    '''
    
    name = 'ActivityScorer'
    
    def __init__(
        self, 
        scoring_units: Dict[str, ScoringUnit],
        units_reduction: Reduction = 'mean',
        layer_reduction: Reduction = 'mean',
        units_map:    UnitsMapping = lambda x: x
    ) -> None:
        
        
        # Units used for scoring
        self._scoring_units = scoring_units
        
        # Units reducing function
        units_reduce: UnitsReduction = partial(
            self._units_reducing,
            # The reduce function is applies the input one on the first dimension
            reduce = partial(
                reduce, 
                pattern='b u -> b',
                reduction=units_reduction
            )
        )

        layer_reduce: LayerReduction = partial(
            self._dict_values_reduction,
            reduction=layer_reduction
        )
        
        super().__init__(
            units_reduce=units_reduce, 
            layer_reduce=layer_reduce,
            units_map=units_map
        )
    
    def _units_reducing(self, state: State, reduce: Callable[[NDArray], NDArray]) -> Dict[str, Score]:
        '''
        Perform aggregation over units across layers

        :param state: State mapping layers to their recorded units activations
        :type state: State
        :param reduce: Reducing function across layers.
        :type reduce: Callable[[NDArray], NDArray]
        :return: _description_
        :rtype: Dict[str, Score]
        '''
        
        self._check_key_consistency(scoring=self.scoring_units.keys(), state=state.keys())
        
        scores = {
            layer: reduce(activations[:, self.scoring_units[layer] if self.scoring_units[layer] else slice(None)])
            for layer, activations in state.items()
            if layer in self.scoring_units
        }
        
        return scores    
    
    @property
    def scoring_units(self) -> Dict[str, ScoringUnit]:
        ''' How many units involved in optimization '''

        return self._scoring_units
    
    
class WeightedPairSimilarityScorer(Scorer):
    '''
    This scorer computes weighted similarities (negative distances)
    between groups of subject states. Weights can either be positive
    or negative. Groups are defined via a grouping function. 
    '''
    
    name = 'WeightedPairSimilarityScorer'

    # TODO: We are considering all recorded neurons
    def __init__(
        self,
        signature : Dict[str, float],
        trg_neurons: Dict[str, ScoringUnit], 
        metric : _MetricKind = 'euclidean',
        dist_reduce_fn  : Callable[[NDArray], NDArray] | None = None,
        layer_reduce_fn : Callable[[NDArray], NDArray] | None = None
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
        dist_reduce_fn  = default(dist_reduce_fn,  np.mean)
        layer_reduce_fn = default(layer_reduce_fn, partial(np.mean, axis=0))
        
        # If similarity function is not given use euclidean distance as default
        self._metric = partial(pdist, metric=metric)
        self._metric_name = metric
        
        criterion: UnitsReduction = partial(
            self._score,
            reduce_fn=dist_reduce_fn,
            neuron_targets=trg_neurons,   
        )
        aggregate: LayerReduction = partial(
            self._dotprod,
            signature=signature,
            reduce_fn=layer_reduce_fn    
        )
        
        self._signature = signature
        
        self._trg_neurons = trg_neurons

        super().__init__(
            units_reduce=criterion,
            layer_reduce=aggregate
        )
        
    def __str__(self) -> str:        
        return f'WeightedPairSimilarityScorer[metric: {self._metric_name}; signature: {self._signature}]'

    @property
    def n_scoring_units(self) -> int:
        ''' How many units involved in optimization '''
        return 0

    def _score(
        self,
        state : State,
        reduce_fn : Callable[[NDArray], NDArray],
        neuron_targets : Dict[str, ScoringUnit],
    ) -> Dict[str, NDArray]:
        
        scores = {
            layer: np.array([
                reduce_fn(-self._metric(group))
                for group in activations[..., neuron_targets[layer] if neuron_targets[layer] else slice(None)]
            ])
            for layer, activations in state.items()
            if layer in neuron_targets
        }
        
        return scores

    def _dotprod(
        self,
        state : Dict[str, Score],
        signature : Dict[str, float],
        reduce_fn : Callable[[NDArray], NDArray],     
    ) -> Score:
        return cast(
            Score,
            reduce_fn(
                np.stack([v * state[k] for k, v in signature.items()])
            )
        )
        
    @property
    def scoring_units(self) -> Dict[str, ScoringUnit]:
        ''' How many units involved in optimization '''
        return self._trg_neurons

