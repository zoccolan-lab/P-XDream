import numpy as np
from abc import abstractmethod

from utils import default
from utils import lazydefault

from typing import Callable, Dict, Tuple 
from numpy.typing import NDArray

ObjectiveFunction = Callable[[NDArray | Dict[str, NDArray]], float]
SubjectState = NDArray | Dict[str, NDArray]

class Optimizer:
    '''
    Base class for generic optimizer, which keeps track of current
    parameter set (codes) and defines the abstract `step()` method 
    which every concrete instantiation should implement.
    '''
    
    def __init__(
        self,
        objective_fn : ObjectiveFunction,
        states_space : None | Dict[int | str, Tuple[float | None, float | None]] = None,
        states_shape : None | int | Tuple[int, ...] = None,
        random_state : None | int = None,
        random_distr : None | str = None,
    ) -> None:
        '''
        Create a  gradient-free optimizer by specifying its objective
        function and (optional) state space conditions.
        
        :param objective_fn: Function that accepts a current state
            (type SubjectState) and computes the associated score
            (a single float scalar number)
        :type objective_fn: ObjectiveFunction (Callable with specific
            signature)
        :param states_space: Dictionary specifying the optimization
            domain where each entry is a direction and corresponding
            values are the min-max acceptable values along that dir
        :type states_spaces: Dictionary of int|str -> Tuple[int, int]
        :param states_shape: Shape specifying the dimensionality of
            the optimization domain. If states_space is specified it
            supersedes the states_shape as shape is inferred from the
            dictionary dimensionality.
        :type states_shape: int or tuple of ints
        '''
        assert states_shape is not None or states_space is not None,\
            'Either states_shape or states_space must be specified'
        
        if isinstance(states_shape, int):
            states_shape = (states_shape,)
        
        self.obj_fn = objective_fn
        self._space = lazydefault(states_space, lambda : {i : (None, None) for i in range(len(states_shape))}) # type: ignore
        self._shape = lazydefault(states_shape, lambda : (len(states_space),))                                 # type: ignore
        
        self._param = []
        self._score = []
        self._distr = random_distr
        
        # Initialize the internal random number generator for reproducibility
        self._rng = np.random.default_rng(random_state)
        
    def init(self, init_cond : str | NDArray = 'normal', **kwargs) -> None:
        '''
        Initialize the optimizer parameters. If initial parameters
        are provided they should have matching dimensionality as 
        expected by provided states shape, otherwise they are sampled
        randomly (distribution can be specified via a name string)
        
        :param init_cond: initial condition for optimizations
        :type init_cond: either string (name of distribution
            to use, default: normal) or numpy array
            
        :param kwargs: parameters that are passed to the random
            generator to sample from the chosen distribution
            (e.g. loc=0, scale=1 for the choice init_cond=normal)
        '''
        
        if isinstance(init_cond, np.ndarray):
            assert init_cond.shape == self._shape,\
                'Provided initial condition does not match expected shape'
            self._param = init_cond.copy()
        else:
            self._distr = init_cond
            self._param = self.rnd_sample(**kwargs)                
    
    @abstractmethod
    def step(self, states : SubjectState) -> NDArray:
        '''
        Abstract step method. The `step()` method collects the set of
        old states from which it obtains the set of new scores via the
        objective function and updates the internal parameters to produce
        a new set of states that would hopefully increase future scores
        
        :param states: Set of current observables (e.g. the activations
            of a hidden population in a neural network)
        :type states: Numpy array or dictionary indexed by string (e.g.
            layer names) and corresponding observables as Numpy array
            
        :returns: Set of new parameters (codes) to be used to improve
            future states scores
        :rtype: Numpy array  
        '''
        raise NotImplementedError('Optimizer is abstract. Use concrete implementations')
    
    @property
    def rnd_sample(self) -> Callable:
        match self._distr:
            case 'normal': return self._rng.normal
            case 'gumbel': return self._rng.gumbel
            case 'laplace': return self._rng.laplace
            case 'logistic': return self._rng.logistic
            case _: raise ValueError(f'Unrecognized distribution: {self._distr}')
    
    @property
    def stats(self) -> Dict[str, NDArray]:
        best_idx = np.argmax(self._score)
        
        return {
            'best_score' : self._score[best_idx],
            'best_param' : self._param[best_idx],
            'curr_score' : self._score[-1],
            'curr_param' : self._param[-1],
        }
        
    @property
    def scores(self) -> NDArray:
        return self._score[-1]
    
    @property
    def param(self) -> NDArray:
        return self._param[-1]
    
class GeneticOptimizer(Optimizer):
    '''
    '''
    
    def __init__(
        self,
        objective_fn: ObjectiveFunction,
        states_space : None | Dict[int | str, Tuple[float | None, float | None]] = None,
        states_shape : None | int | Tuple[int, ...] = None,
        random_state : None | int = None,
        random_distr : None | str = None,
        mutation_size : float = 0.1,
        mutation_rate : float = 0.3,
        population_size : int = 50,
        num_parents : int = 2,
    ) -> None:
        '''
        
        '''
        
        super().__init__(
            objective_fn,
            states_space,
            states_shape,
            random_state, 
            random_distr
        )
        
        self.num_parents = num_parents
        self.mutation_size = mutation_size
        self.mutation_rate = mutation_rate
        self.population_size = population_size
    
    def step(self, states : NDArray | Dict[str, NDArray]) -> NDArray:
        pass
    
    def _mutate(self) -> None:
        pass
    
    def _breed(self, num_children : int | None = None) -> None:
        num_children = default(num_children, self.population_size)
        
        # Select the breeding family based on the fitness of each parent
        # NOTE: The same parent can occur more than once in each family
        parents = self._rng.choice(
            self.population_size,
            size=(num_children, self.num_parents),
            p=self.scores,
            replace=True,
        )
        
        # Identify which parent contributes which genes to each child
        parentage = self._rng.choice(self.num_parents, size=(num_children, *self._shape), replace=True)
        
        children = np.empty(shape=(num_children, *self._shape))
        for c, (child, family, lineage) in enumerate(zip(children, parents, parentage)):
            for parent in family:
                genes = lineage == parent
                child[genes] = self.param[parent][genes]