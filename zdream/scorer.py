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


    def __call__(self, states : States) -> Scores:
        '''
        Compute the subject scores given a subject state by 
        using the proper mapping and reducing functions.

        :param states: Tuple containing the subject state.
        :type states: State
        :return: Tuple containing the scores associated to input stimuli.
        :rtype: Scores
        '''
        
        # if current_iter is not None:
        #     self.current_iter = current_iter

        # 1. Mapping activations
        # state_mapped: States  = {
        #     layer: self._units_map(act.copy())
        #     for layer, act in states.items()
        # }
        # 
        # # 2. Performing units reduction
        # layer_scores: Dict[str, Scores] = self._units_reduce(state_mapped)
        # self.layer_scores = layer_scores
        # 
        # # 3. Performing layer reduction
        # scores = self._layer_reduce(layer_scores)
        
        scores = self.layer_reduce(states=states)

        return scores
    
    def states_mapping(self, states: States) -> States:
        
        return {
            layer: self._units_map(act.copy())
            for layer, act in states.items()
        }
        
    def unit_reduction(self, states: States) -> Dict[str, Scores]:
        
        return self._units_reduce(self.states_mapping(states))
    
    def layer_reduce(self, states: States) -> Scores:
        
        return self._layer_reduce(self.unit_reduction(states))
    
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

class PairDistanceScorer(Scorer):
    """
    Class implementing a score associated to a pair of states, using their layer-wise distance.
    The class implements the units reduction using an arbitrary distance function.
    
    It doesn't implement any specific layer reduction, which remains general and in case
    to be implemented in possible subclasses.
    """
    
    
    def __init__(
        self,
        scoring_units : Dict[str, ScoringUnit],
        layer_reduce  : LayerReduction,
        metric        : _MetricKind                  = 'euclidean',
        dist_reduce   : Callable[[NDArray], NDArray] = np.mean,
    ):
        """
        Initialize the PairDistanceScorer object.

        :param scoring_units: Target neurons for each layer.
        :type scoring_units: Dict[str, ScoringUnit]
        :param layer_reduce: Function to reduce layer scores.
        :type layer_reduce: LayerReduction
        :param metric: Distance metric to be used in the similarity computation, defaults to 'euclidean'.
        :type metric: _MetricKind, optional
        :param dist_reduce: Function to reduce the distance across units, defaults to the mean.
        :type dist_reduce: Callable[[NDArray], NDArray], optional
        """
        
        # Save the input parameters
        self._scoring_units = scoring_units
        self._metric_name = metric
        self._metric = partial(pdist, metric=self._metric_name)
        
        # Define the units reduction function by fixing the distance metric
        units_reduce: UnitsReduction = partial(
            self._distance_reduction,
            reduce=dist_reduce,
            scoring_units=scoring_units
        )
        
        super().__init__(
            units_reduce=units_reduce,
            layer_reduce=layer_reduce
        )
        
    def __str__(self) -> str:
        return  f'{super()}['\
                f'layers: { {layer: len(units) for layer, units in self._scoring_units.items()}  }'\
                f'metric: {self._metric_name}'\
                ']'
        
    def _distance_reduction(
        self,
        state         : States,
        reduce        : Callable[[NDArray], NDArray],
        scoring_units : Dict[str, ScoringUnit],
    ) -> Dict[str, NDArray]:
        '''
        Class used for the unit reduction.
        
        It computes the similarity scores between groups of activations in the subject state.

        :param state: Subject state.
        :type state: State
        :param reduce: Reducing function across groups of activations.
        :type reduce: Callable[[NDArray], NDArray]
        :param scoring_units: Target neurons for each layer.
        :type scoring_units: Dict[str, ScoringUnit]
        :return: Similarity scores between groups of activations.
        :rtype: Dict[str, NDArray]
        '''
        
        scores = {
            layer: np.array([
                reduce(-self._metric(group))
                for group in activations[..., 
                    scoring_units[layer]    # Specified scoring units for that layer
                    if scoring_units[layer]
                    else slice(None)      # In the case of no units specified, all units are considered
                ]
            ])
            for layer, activations in state.items()
            if layer in scoring_units
        }
        
        return scores
    
    @property
    def scoring_units(self) -> Dict[str, ScoringUnit]:
        ''' How many units involved in optimization '''
        return self._scoring_units

    
class WeightedPairDistanceScorer(PairDistanceScorer):
    """
    Class implementing a score associated to a pair of states,
    by combining their layer-wise activation distance in a weighted fashion.
    """
    
    def __init__(
        self,
        layer_weights : Dict[str, float],
        scoring_units : Dict[str, ScoringUnit],
        metric        : _MetricKind = 'euclidean',
        dist_reduce   : Callable[[NDArray], NDArray] = np.mean,
        layer_reduce  : Callable[[NDArray], NDArray] = np.mean
    ):
        """
        Initialize the WeightedPairDistanceScorer object.

        :param layer_weights: Weights to be used for the weighted distance computation.
        :type layer_weights: Dict[str, float]
        :param scoring_units: Target neurons for each layer.
        :type scoring_units: Dict[str, ScoringUnit]
        :param metric: Distance metric to be used in the similarity computation, defaults to 'euclidean'.
        :type metric: _MetricKind, optional
        :param dist_reduce: Function to reduce the distance across units, defaults to the mean.
        :type dist_reduce: Callable[[NDArray], NDArray], optional
        :param layer_reduce: Function to reduce layer scores, defaults to the mean.
        :type layer_reduce: Callable[[NDArray], NDArray], optional
        """
        
        self._layer_weights = layer_weights
        
        layer_reduce_: LayerReduction = partial(
            self._dotprod,
            weights=self._layer_weights,
            reduce=layer_reduce
        )
        
        super().__init__(
            scoring_units=scoring_units,
            layer_reduce=layer_reduce_,
            metric=metric,
            dist_reduce=dist_reduce
        )
        
    def __str__(self) -> str:
        return  f'{str(super())[:-1]}; '\
                f'weights: {self._layer_weights}, '\
                f']'
    
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
                np.stack([v * state[k] for k, v in weights.items()])
            )
        )

class ParetoReferencePairDistanceScorer(PairDistanceScorer):
    '''
    The scorer computes the similarity between the subject state and a reference state involving multiple layers.
    It associates to each layer a weight that in a pareto front optimization to assign a score to each state.
    
    The score is assigned such that:
    - units in an higher pareto front have an higher score
    - units in the same pareto front use a random internal rank 
    
    Since we using pairs of states but a reference, a state preprocessing is needed 
    to create a pair of states with the reference as the second state.
    '''
    
    name = 'WeightedPairSimilarityScorer'

    def __init__(
        self,
        layer_weights : Dict[str, float],
        scoring_units : Dict[str, ScoringUnit], 
        reference     : Dict[str, NDArray],
        metric        : _MetricKind = 'euclidean',
        dist_reduce   : Callable[[NDArray], NDArray] = np.mean,
        bounds        : Dict[str, Callable[[float], bool]] | None = None
    ) -> None: 
        '''
        Initialize the scorer.
        
        :param scoring_units: Dictionary mapping each recorded layer to the corresponding 
            weight to be used in the final aggregation step. 
            Positive weights (> 0) denote desired similarity, 
            while negative weights (< 0) denote desired dissimilarity.
        :type scoring_units: Dict[str, float]
        :param trg_neurons: Dictionary mapping each layer to the indices of the units to be scored in that layer.
        :type trg_neurons: Dict[str, ScoringUnit]
        :param reference: Dictionary describing the states of the target reference. The state is supposed to
            describe all the layers used for recordings.
        :param metric: Distance metric to be used in the similarity computation.
        :type metric: string
        :param dist_reduce: Function to reduce the distance across units, defaults to the mean.
        :type dist_reduce: Callable[[NDArray], NDArray], optional
        :param bounds: TODO @DonTau. If specified, the bounds must be specified for all the layers in the reference.
        :type bounds: Dict[str, Callable[[float], bool]] | None, defaults to None
        
        '''
        
        # Sanity Check - Check if all the layers in the reference are specified in the bounds
        if bounds and not all(k in reference for k in bounds):
            
            not_specified = set(reference.keys()).difference(set(bounds.keys()))
            
            raise ValueError(
                f'Bounds must be specified for all the layers in the reference', 
                f'but the following layers are not specified: {not_specified}'
            )
        
        # Save the input parameters
        self._reference     = reference
        self._bounds        = bounds
        self._layer_weights = layer_weights
        
        # Define reducing function across layers using the pareto front
        layer_reduce_: LayerReduction = partial(
            self._best_pareto,
            weights=layer_weights
        )
        
        # Initialize the parent class providing the two reductions
        super().__init__(
            scoring_units=scoring_units,
            layer_reduce=layer_reduce_,
            metric=metric,
            dist_reduce=dist_reduce
        )
        
    def _preprocess_states(self, states: States) -> States:
        """ 
        Preprocess the states by appending the reference to the states for each layer
        
        :layer states: The states to preprocess
        :type states: States
        :return: The preprocessed states
        :rtype: States
        """
        
        # New states
        states_ = {}
        
        # Add the reference to the states
        for layer in states:
            
            if layer in self._reference:
                
                # Repeat the reference as many times as the states
                repeated_b_array = np.repeat(self._reference[layer], states[layer].shape[0], axis=0)
                
                # Stack the states and the reference
                states_[layer]   = np.stack((states[layer], repeated_b_array), axis=1)
                
        return states_
    
    def __call__(self, states: States):
        """ 
        STUB to the parent class using preprocessed states
        """
        
        states_preprocess = self._preprocess_states(states)
        
        return super().__call__(states=states_preprocess)
        
    # --- STRING REPRESENTATION ---
    
    def __str__(self) -> str:        
        ''' Return a string representation of the scorer including also the signature '''
        return  f'{str(super())[:-1]}; '\
                f'weights: {self._layer_weights}, '\
                f'reference: { {layer: state.shape for layer, state in self._reference.items()} }, '\
                f']'
    
    # --- PARETO ---
    
    @staticmethod
    def pareto_front(
        state            : Dict[str, Scores],
        weights          : List[float], 
        first_front_only : bool = False
    ):
        """
        Compute the pareto front of the given state.
        
        :param state: The state to compute the pareto front
        :type state: Dict[str, Scores]
        :param weights: The weights to use for the pareto front
        :type weights: List[float]
        :param first_front_only: Whether to return only the first front
        :type first_front_only: bool
        """
        
        # Create 
        creator.create("FitnessMulti", base.Fitness, weights=tuple(weights))
        creator.create("Individual", list, fitness=creator.FitnessMulti) # type: ignore
        layers = list(state.keys()) 
        
        # Create the population
        individuals = [
            creator.Individual([  # type: ignore
                state[layer][i] 
                for layer in layers
            ]) 
            for i, _ in enumerate(state[layers[0]])
        ]
        
        # TODO @DonTau
        for individual_id, individual in enumerate(individuals):
            individual.fitness.values = tuple(individual)
            individual.id = individual_id
        
        # Compute the pareto front
        fronts = tools.sortNondominated(
            individuals=individuals, 
            k=len(individuals), 
            first_front_only=first_front_only
        )
        
        scores = np.zeros([len(individuals)])
        
        # Assign the scores
        for front_id, front in enumerate(fronts):
            for individual_id, individual in enumerate(front):
                scores[individual.id] = np.abs(front_id - (len(fronts)))
        
        # TODO @DonTau
        coordinates_p1 = np.where(scores == np.max(scores))
        
        return scores, coordinates_p1[0]
        
    def _best_pareto(
        self,
        state     : Dict[str, Scores],
        weights   : Dict[str, float]
    )-> Scores:
        """
        The function acts as a layer reduction function.
        
        It computes the best pareto front for the given state 
        and assign a score to each individual in the population

        :param state: State to compute the pareto front
        :type state: Dict[str, Scores]
        :param weights: Weights to use for the pareto front
        :type weights: Dict[str, float]
        :return: The scores for the given state
        :rtype: Scores
        """
        
        # Apply bound constraints
        state_dup = self._bound_constraints(state)
        
        pf_scores, coordinates_p1 = self.pareto_front(state_dup, weights = [v for v in weights.values()])
        self.coordinates_p1 = coordinates_p1
        rand_scores = np.random.rand(pf_scores.shape[0])
        scores = (pf_scores+1)*(max(rand_scores)+1)+rand_scores
        
        return scores
    
    # --- BOUND CONSTRAINTS ---
    
    def _bound_constraints(self, state: States) -> States:
        """
        Apply bound constraints to the state, pushing to minus infinity
        the values that do not satisfy the constraints.
        
        In the case no bounds are specified, the state is returned as it is.

        :param state: The state to apply the bound constraints
        :type state: States
        :return: The state with the bound constraints
        :rtype: States
        """
        
        if self._bounds is None:
            return state
        
        state_dup = {
            layer: np.array([
                individual_state if self._bounds[layer](individual_state) else -float('inf') 
                for individual_state in layer_state
            ])
            for layer, layer_state in state.items()
        }
        
        return state_dup
    
    def bound_constraints(self, state: States) -> States:
        """ 
        It is a stub to the private method using the preprocessed states 
        It is supposed to be used externally to the typical `__call__` pipeline
        """
        return self._bound_constraints(state=self._preprocess_states(state))





