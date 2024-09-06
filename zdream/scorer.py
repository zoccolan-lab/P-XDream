'''
This file implements the Scorer, a class for computing stimuli scores given subject states.
A Scorer perform mapping and reducing operation across units and layers and returns a float score for each state.

The file implements three main classes:
1. TargetScorer: A class for computing the MSE between the subject state and a fixed target.
2. ActivityScorer: A class for computing the activity of units in different layers.
3. WeightedPairSimilarityScorer: A class for computing weighted similarities between groups of subject states.
'''

from _collections_abc import dict_keys
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, cast, Literal

import numpy as np
from numpy.typing import NDArray
from einops import rearrange, reduce
from einops.einops import Reduction
from scipy.spatial.distance import pdist, squareform
from deap import base, creator, tools


from .utils.types import LayerReduction, ScoringUnit, Scores, States, UnitsReduction, UnitsMapping
from .utils.misc import default, minmax_norm

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

    The scoring process is break up in three steps:

    1. Mapping activations - a function for activation element-wise mapping for each layer.
    2. Reducing unit  - a function that reduces multiple units of a single layer into
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
        Initialize a scorer by defining the three main transformations.

        :param units_reduce: Function to reduce units activations.
        :type units_reduce: UnitsReduction
        :param layer_reduce: Function to reduce layer scores.
        :type layer_reduce: LayerReduction
        :param units_map: Function to map units activations.
        :type units_map: UnitsMapping
        '''

        # Save input transformation functions
        self._units_map    : UnitsMapping   = units_map
        self._layer_reduce : LayerReduction = layer_reduce
        self._units_reduce : UnitsReduction = units_reduce


    def __call__(self, states : States, current_iter = None) -> Scores | None:
        '''
        Compute the subject scores given a subject state by 
        using the proper mapping and reducing functions.

        :param states: Tuple containing the subject state.
        :type states: State
        :return: Tuple containing the scores associated to input stimuli.
        :rtype: Scores
        '''
        if current_iter is not None:
            self.current_iter = current_iter

        # 1. Mapping activations
        state_mapped: States  = {
            layer: self._units_map(act.copy())
            for layer, act in states.items()
        }
        
        # 2. Performing units reduction
        layer_scores: Dict[str, Scores] = self._units_reduce(state_mapped)
        self.layer_scores = layer_scores
        
        # 3. Performing layer reduction
        scores = self._layer_reduce(layer_scores)

        return scores
    
    # --- STRING REPRESENTATION ---
        
    def __str__(self) -> str:
        ''' Return a string representation of the scorer. '''
        
        # Get the dimensions of the target for different layers
        dims = ", ".join([f"{k}: {len(v)} units" for k, v in self.scoring_units.items()])
        
        return f'{self._NAME}[target size: ({dims})]'
    
    def __repr__(self) -> str: return str(self)
    ''' Return a string representation of the scorer. '''
    

    # --- PROPERTIES ---

    @property
    @abstractmethod
    def scoring_units(self) -> Dict[str, ScoringUnit]:
        '''
        Returns the scoring units associated to each layer.
        Units index refers to activations in the layer.

        :return: Scoring units across layers.
        :rtype: Dict[str, ScoringUnit]
        '''
        pass

    @property
    def n_scoring_units(self) -> int:
        ''' 
        Returns the total number of scoring units

        :return: Total number of scoring units.
        :rtype: int
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
        Check subject state to contain at least the target keys for computing the scores of interest. 
        Additional keys in subject (i.e. layers only for recording) will be ignored by the scorer.

        :param scoring: Scoring layers.
        :type scoring: dict_keys
        :param state: State layers.
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
        target          : States,
        layer_reduction : Reduction = 'mean'
    ) -> None:
        '''
        Initialize the MSE scorer as a neuron which preferred stimulus is fixed 
        as a target state. The MSE is computed as the mean squared error between
        the subject state and the fixed target.

        :param target: Fixed target state.
        :type target: State
        :param layer_reduction: Reducing function across layer scores, defaults to the mean.
        :type layer_reduction: ScoringReducing
        '''
        
        self._target : States = target
        
        # The mapping function is the MSE between the 
        # subject state and the fixed target. 
        units_reduce: UnitsReduction = partial(self._mse_reduction, target=self._target)

        # The layer reduction function is the input reducing function across layers
        layer_reduce: LayerReduction = partial(
            self._dict_values_reduction,
            reduction=layer_reduction
        )
        
        # Initialize the parent class providing the two reductions
        super().__init__(
            units_reduce=units_reduce,
            layer_reduce=layer_reduce
        )
        
        
    # --- MSE ---    
    
    def _mse_reduction(self, state: States, target: States) -> Dict[str, Scores]:
        '''
        Compute the MSE between the subject state and the fixed target.

        :param state: Subject state.
        :type state: State
        :param target: Fixed target.
        :type target: State
        :return: MSE between target and state.
        :rtype: Dict[str, Score]
        '''

        # Check for layer name consistency
        self._check_key_consistency(scoring=target.keys(), state=state.keys())
        
        # Compute the MSE between the subject state and the fixed target
        # NOTE: The minus sign is used to have the best score (0 distance) 
        #       as the maximum value, as the employed optimizer is a maximizer.
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
    
    # --- PROPERTIES ---
    
    @property
    def target(self) -> States: return self._target
    ''' Fixed target state for the MSE scorer. '''
    
    @property
    def scoring_units(self) -> Dict[str, ScoringUnit]:
        '''
        Return the target units for each layer as the product of the target shapes.

        :return: Target units for each layer.
        :rtype: Dict[str, ScoringUnit]
        '''

        return {k: list(range(np.prod(v.shape))) for k, v in self._target.items()}

        ...
class ActivityScorer(Scorer):
    '''
    A Scorer class to compute a single cross aggregating across multiple units in different layers.
    
    This class is used to score the activity of units in different layers. 
    
    It takes a dictionary of scoring units, which maps layer names to the indices of the units to be scored in that layer. 
    It also supports different reduction methods for aggregating the scores across units and layers.
    '''
    
    name = 'ActivityScorer'
    
    def __init__(
        self, 
        scoring_units   : Dict[str, ScoringUnit],
        units_reduction : Reduction = 'mean',
        layer_reduction : Reduction = 'mean',
        units_map       : UnitsMapping = lambda x: x
    ) -> None:
        '''
        Initialize the Scorer object.

        :param scoring_units: Units to score for each layer.
        :type scoring_units: Dict[str, ScoringUnit]
        :param units_reduction: Reduction method to be applied to the scoring units. Defaults to 'mean'.
        :type units_reduction:  Reduction, optional
        :param layer_reduction: Reduction method to be applied to the layers. Defaults to 'mean'.
        :type layer_reduction: Reduction, optional
        :param units_map: Mapping function for the scoring units. Defaults to identity function.
        :type units_map: UnitsMapping, optional
        '''
        
        # Save units involved in scoring
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

        # Layer reducing function
        layer_reduce: LayerReduction = partial(
            self._dict_values_reduction,
            reduction=layer_reduction
        )
        
        # Initialize the parent class providing the three main functions
        super().__init__(
            units_reduce=units_reduce, 
            layer_reduce=layer_reduce,
            units_map=units_map
        )
        
    # --- UNITS REDUCTION ----
    
    def _units_reducing(self, state: States, reduce: Callable[[NDArray], NDArray]) -> Dict[str, Scores]:
        '''
        Perform aggregation over units across layers

        :param state: State mapping layers to their recorded units activations
        :type state: State
        :param reduce: Reducing function across layers.
        :type reduce: Callable[[NDArray], NDArray]
        :return: _description_
        :rtype: Dict[str, Score]
        '''
        
        # Check for layer name consistency
        self._check_key_consistency(scoring=self.scoring_units.keys(), state=state.keys())
        
        # Compute the scores for each layer
        scores = {
            
            # Compute the reduction of the units activations
            # supporting all-units encoding
            layer: reduce(
                activations[:,
                    self.scoring_units[layer]     # Specified scoring units for that layer if any 
                    if self.scoring_units[layer]
                    else slice(None)              # In the case of no units specified, all units are considered
                ]
            )
            for layer, activations in state.items()
            if layer in self.scoring_units
            
        }
        
        return scores 
    
    # --- PROPERTIES ---
    
    @property
    def scoring_units(self) -> Dict[str, ScoringUnit]:
        '''
        Return the scoring units associated to each layer.

        :return: Scoring units across layers.
        :rtype: Dict[str, ScoringUnit]
        '''

        return self._scoring_units


class WeightedPairSimilarityScorer(Scorer):
    '''
    The scorer computes weighted similarities between groups of subject states.
    It uses negative distances so that identic object will have the maximum score of 0.
    in a logic of optimizer maximization.
    
    Each layer is associated with a weight, that can be either positive or negative,
    which is used to compute the final score.
    '''
    
    name = 'WeightedPairSimilarityScorer'

    def __init__(
        self,
        layer_weights : Dict[str, float],
        trg_neurons   : Dict[str, ScoringUnit], 
        metric        : _MetricKind = 'euclidean',
        dist_reduce   : Callable[[NDArray], NDArray] | None = None,
        layer_reduce  : Callable[[NDArray], NDArray] | None = None,
        reference     : dict[str, Any] | None = None,
        bounds        : Dict[str, Callable[[float], bool]] | None = None
        
    ) -> None: 
        '''


        :param layer_weights: Dictionary mapping each recorded layer 
            (dict key) the corresponding weight (a float) to be used in the 
            final aggregation step. Positive weights (> 0) denote desired similarity, 
            while negative weights (< 0) denote desired dissimilarity.
        :type layer_weights: Dict[str, float]
        :param metric: Distance metric to be used in the similarity computation.
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
        
        # Save input parameters
        self._signature   = layer_weights
        self._trg_neurons = trg_neurons
        self._bounds      = bounds
        
        # If grouping function is not given use even-odd split as default
        dist_reduce  = default(dist_reduce,  np.mean)
        layer_reduce = default(layer_reduce, partial(np.mean, axis=0))
        
        # If similarity function is not given use euclidean distance as default
        self._metric = partial(pdist, metric=metric)
        self._metric_name = metric
        
        # Define reducing function across units using the given distance function
        units_reduce: UnitsReduction = partial(
            self._score,
            reduce=dist_reduce,
            trg_neurons=trg_neurons,   
        )
        
        # Define reducing function across layers using the given distance 
        # function and the layer weights
        layer_reduce_: LayerReduction = partial(
            #self._dotprod, 
            self._best_pareto,
            weights=layer_weights,
            reduce=layer_reduce    
        )
        
        # Initialize the parent class providing the two reductions
        super().__init__(
            units_reduce=units_reduce,
            layer_reduce=layer_reduce_
        )
        
    # --- STRING REPRESENTATION ---
    
    def __str__(self) -> str:        
        ''' Return a string representation of the scorer including also the signature '''
        return f'{super()}[signature: {self._signature}]'
    
    # --- REDUCTIONS ---

    def _score(
        self,
        state       : States,
        reduce      : Callable[[NDArray], NDArray],
        trg_neurons : Dict[str, ScoringUnit],
    ) -> Dict[str, NDArray]:
        '''
        Compute the similarity scores between groups of activations in the subject state.

        :param state: Subject state.
        :type state: State
        :param reduce: Reducing function across groups of activations.
        :type reduce: Callable[[NDArray], NDArray]
        :param trg_neurons: Target neurons for each layer.
        :type trg_neurons: Dict[str, ScoringUnit]
        :return: Similarity scores between groups of activations.
        :rtype: Dict[str, NDArray]
        '''
        
        scores = {
            layer: np.array([
                reduce(-self._metric(group))
                for group in activations[..., 
                    trg_neurons[layer]    # Specified scoring units for that layer
                    if trg_neurons[layer]
                    else slice(None)      # In the case of no units specified, all units are considered
                ]
            ])
            for layer, activations in state.items()
            if layer in trg_neurons
        }
        
        return scores

    def _dotprod(
        self,
        state     : Dict[str, Scores],
        weights   : Dict[str, float],
        reduce    : Callable[[NDArray], NDArray],     
    ) -> Scores:
        '''
        Compute the dot product of the state and weights, and reduce the result using the given reduce function.

        :param state: A dictionary containing the state values.
        :type state: Dict[str, Score]
        :param weights: A dictionary containing the weight values.
        :type weights: Dict[str, float]
        :param reduce: A function used to reduce the result of the dot product.
        :type reduce: Callable[[NDArray], NDArray]
        :return: The result of the dot product after reducing.
        :rtype: Score
        '''
        return cast(
            Scores,
            reduce(
                # Multiply each layer score by the corresponding weight
                #np.stack([v * minmax_norm(state[k]) for k, v in weights.items()])
                np.stack([v * state[k] for k, v in weights.items()])
            )
        )
        
    def _best_pareto(
        self,
        state     : Dict[str, Scores],
        weights   : Dict[str, float],
        reduce    : Callable[[NDArray], NDArray], 
    )-> Scores:
        
        creator.create("FitnessMulti", base.Fitness, weights=tuple([v for _,v in weights.items()]))  # Max a, Min b
        creator.create("Individual", list, fitness=creator.FitnessMulti); s_keys = list(state.keys())
        #state = {k: minmax_norm(state[k]) for k in s_keys}
        state_dup = {}
        for key in s_keys:
            state_dup[key] = [x if self._bounds[key](x) else -float('inf') for x in state[key]]
            valid_values_count = sum(1 for value in state_dup[key] if value != float('-inf'))
            if valid_values_count < 10 and self.current_iter > 5:
                return None

        pop = [creator.Individual([state_dup[k][i] for k in s_keys]) for i in range(len(state_dup[s_keys[0]]))]
        #pop = [creator.Individual([state[k][i] for k in s_keys]) for i in range(len(state[s_keys[0]]))] #old manual way that works good

        for i,ind in enumerate(pop):
            ind.fitness.values = tuple(ind)
            #ind.fitness.values = tuple(ind) if np.abs(ind[1]) < 3 else tuple([ind[0],-float('inf')])
            #ind.fitness.values = tuple(ind) if np.abs(ind[0]) < 180 else tuple([-float('inf'),ind[1]])
            ind.id = i
        fronts = tools.sortNondominated(pop, len(pop))
        scores = np.zeros([2, len(pop)])
        for f_id, f in enumerate(fronts):
            #dist_f = np.mean(squareform(pdist(np.array(f), metric='euclidean')), axis = 0)
            for i,ind in enumerate(f):
                scores[0, ind.id] = np.abs(f_id - len(fronts))
                #scores[1, ind.id] = 1/(dist_f[i]+0.0001)
        scores[1, :] = np.random.rand(scores.shape[1])
                
        scores = (scores[0,:]+1)*(max(scores[1,:])+1)+scores[1,:]
        
        return cast(Scores, scores)                
        
    
        


    # --- PROPERTIES ---
        
    @property
    def scoring_units(self) -> Dict[str, ScoringUnit]:
        ''' How many units involved in optimization '''
        return self._trg_neurons

