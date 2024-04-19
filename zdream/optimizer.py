from abc import ABC, abstractmethod
from typing import Callable, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax
from cma import CMAEvolutionStrategy

from .utils.model import Codes, Score
from .utils.misc import default
from .message import ZdreamMessage

RandomDistribution = Literal['normal', 'gumbel', 'laplace', 'logistic']
''' 
Name of distributions for random initial codes
'''

class Optimizer(ABC):
    '''
    Abstract class for a generic optimizer.
    It implements a function `step()` to produce a new set of
    codes based on a scoring input.
    '''
    
    '''
    TODO Add back for future implementation
    
    `states_space`: None | Dict[int | str, Tuple[float | None, float | None]] = None
    Dictionary specifying the optimization domain where 
    each entry is a direction and corresponding values
    are the min-max acceptable values along that dir, defaults to None.
    
    `states_shape`: None | int | Tuple[int, ...] = None
    Dictionary specifying the optimization domain where 
    each entry is a direction and corresponding values are 
    the min-max acceptable values along that direction, defaults to None.
    
    if not (states_shape or states_space):
            err_msg = 'Either `states_shape` or `states_space` must be specified, but both weren\'t.'
            raise ValueError(err_msg)
                
        # States shape - int to tuple conversion
        if isinstance(states_shape, int):
            states_shape = (states_shape,)
            
    self._space = lazydefault(states_space, lambda : {i : (None, None) for i in range(len(states_shape))})  # type: ignore
    self._shape = lazydefault(states_shape, lambda : (len(states_space),))                                  # type: ignore
    '''
    
    # --- INIT ---
    
    def __init__(
        self,
        pop_size    : int,
        codes_shape : int | Tuple[int, ...],
        rnd_seed    : None | int = None,
        rnd_distr   : RandomDistribution = 'normal',
        rnd_scale   : float = 1.
    ) -> None:
        '''
        Instantiate a  gradient-free optimizer.

        :param pop_size: Number of initial codes.
        :type pop_size: int
        :param codes_shape: Codes shape. If one dimensional the single dimension supported.
        :type codes_shape: int | Tuple[int, ...]
        :param rnd_seed: Random state for pseudo-random numbers generation.
        :type rnd_seed: None | int, optional
        :param rnd_distr: Nature of the random distribution for initial
                         random codes generation, defaults to `normal`.
        :type rnd_distr: RandomDistribution
        :param rnd_scale: Scale for initial codes generation, defaults to 1.
        :type rnd_scale: float
        '''
        
        # Save initial number of codes
        self._init_n_codes = pop_size
        
        # Codes shape with single dimension cast
        if isinstance(codes_shape, int):
            codes_shape = codes_shape,
        self._codes_shape = codes_shape
        
        # Randomic components
        self._rng        = np.random.default_rng(rnd_seed)
        self._rnd_scale  = rnd_scale
        self._rnd_sample = self._get_rnd_sample(distr  = rnd_distr)
        
        # Last generated codes
        self._codes : None | Codes = None
        
    # --- PROPERTIES ---
        
    @property
    def codes(self) -> Codes:
        '''
        Returns codes produced in the last step.
        
        :return: Last produced codes.
        :rtype: Codes
        '''
        
        # TODO: Codes are optimized in 1 dimension
        # TODO: Handle ravel/unravel using `self._codes_shape`
        # TODO: Ignoring batch dimensions
        
        if self._codes is None:
            err_msg = 'No codes available. Use `init()` method to generate the first codes.'
            raise ValueError(err_msg)
        
        pop_size, *_ = self._codes.shape
        
        codes_ = np.reshape(self._codes, (pop_size, *self._codes_shape))

        return codes_.copy()
    
    
    @property
    def pop_size(self) -> int:
        ''' 
        Number of codes the optimizer is optimizing for.
        NOTE: The number of codes can change dynamically 
            during the optimization process
        '''
        
        try:
            pop_size, *_ = self.codes.shape
            return pop_size
        except ValueError:
            return self._init_n_codes
    
    # --- STEP ---
    
    def step(self, data: Tuple[Score, ZdreamMessage]) -> Tuple[Codes, ZdreamMessage]:
        '''
        Wrapper for actual step implementation in `_step()` that automatizes 
        saving of last generation codes in `_codes`.
        
        :param data: Tuple containing the score associated to each old code and
                     a generic message.
        :type data: Tuple[Score, ZdreamMessage]
        :return: Set of new codes to be used to improve future states scores and a message.
        :rtype: Tuple[Score, ZdreamMessage]
        ''' 
        
        self._codes, msg = self._step(data)

        return self.codes, msg
    
    @abstractmethod
    def _step(self, data: Tuple[Score, ZdreamMessage]) -> Tuple[Codes, ZdreamMessage]:
        '''
        Abstract step method.
        The `step()` method receives the scores associated to the
        last produced codes and uses them to produce a new set of codes.
        
        :param data: Tuple containing the score associated to each old code and
                     a generic message.
        :type data: Tuple[Score, ZdreamMessage]
        :return: Set of new codes to be used to improve future states scores and a message.
        :rtype: Tuple[Score, ZdreamMessage]
        '''        
        
        if self._codes is None:
            err_msg = 'No codes provided, use `init()` to generate first ones. '
            raise ValueError(err_msg)
    
    
    # --- CODE INITIALIZATION ---
    
    def init(
        self, 
        init_codes : NDArray | None = None, 
        **kwargs
    ) -> Codes:
        '''
        Initialize the optimizer codes. 
        
        If initial codes are provided as arrays they should have matching
        dimensionality as expected by the provided states shape,
        otherwise they are randomly sampled.
        
        :param init_codes: Initial codes for optimizations, optional.
        :type init_codes: NDArray | None.
        :param kwargs: Parameters that are passed to the random
                       generator to sample from the chosen distribution
                       (e.g. loc=0, init_cond=normal).
        '''
        
        # Codes provided
        if isinstance(init_codes, np.ndarray):
            
            # Check shape consistency
            exp_shape = (self.pop_size, *self._codes_shape)
            if init_codes.shape != exp_shape:
                err_msg = f'Provided initial codes have shape: {init_codes.shape}, '\
                          f'do not match expected shape {exp_shape}'
                raise Exception(err_msg)
            
            # Use input codes as first codes
            self._codes = init_codes
            
        # Codes not provided: random generation
        else:
            
            # Generate codes using specified random distribution
            self._codes = self._rnd_codes_generation(**kwargs)

        return self.codes
    
    def _rnd_codes_generation(self, **kwargs):
        
        return self._rnd_sample(
            size=(self._init_n_codes, *self._codes_shape),
            scale=self._rnd_scale,
            **kwargs
        )
        
    
    def _get_rnd_sample(
        self,
        distr : RandomDistribution = 'normal'
    ) -> Callable:
        '''
        Uses the distribution input attributes to return 
        the specific distribution function.
        
        :param distr: Random distribution type.
        :type distr: RandomDistribution
        :param scale: Random distribution scale.
        :type scale: float.
        :return: Distribution function.
        :rtype: Callable
        '''
        
        match distr:
            case 'normal':   return self._rng.normal
            case 'gumbel':   return self._rng.gumbel
            case 'laplace':  return self._rng.laplace
            case 'logistic': return self._rng.logistic
            case _: raise ValueError(f'Unrecognized distribution: {distr}')
    
class GeneticOptimizer(Optimizer):
    '''
    Optimizer that implements a genetic optimization strategy.
    
    In particular these optimizer devise a population of candidate
    solutions (set of parameters) and iteratively improves the
    given objective function via the following heuristics:
    
    - The top_k performing solution are left unaltered
    - The rest of the population pool are recombined to produce novel
      candidate solutions via breeding and random mutations
    - The n_parents contributing to a single offspring are selected
      via importance sampling based on parents fitness scores
    - Mutations rate and sizes can be adjusted independently 
    '''
    
    def __init__(
        self,
        codes_shape  : int | Tuple[int, ...],
        rnd_seed     : None | int = None,
        rnd_distr    : RandomDistribution = 'normal',
        rnd_scale    : float = 1.,
        pop_size     : int   = 50,
        mut_size     : float = 0.1,
        mut_rate     : float = 0.3,
        n_parents    : int   = 2,
        allow_clones : bool  = False,
        topk         : int   = 2,
        temp         : float = 1.,
        temp_factor  : float = 1.,
    ) -> None:
        '''
        Initialize a new GeneticOptimizer
        
        :param pop_size: Number of initial codes.
        :type pop_size: int
        :param codes_shape: Codes shape. If one dimensional providing the single dimension is supported.
        :type codes_shape: int | Tuple[int, ...]
        :param rnd_seed: Random state for pseudo-random numbers generation.
        :type rnd_seed: None | int, optional
        :param rnd_distr: Nature of the random distribution for initial
                          random codes generation, defaults to `normal`.
        :type rnd_distr: RandomDistribution
        :param rnd_scale: Scale for initial codes generation, defaults to 1.
        :type rnd_scale: float
        :param pop_size: Number of codes in the population, defaults to 50
        :type pop_size: int, optional
        :param mut_size: Probability of single-point mutation, defaults to 0.3
        :type mut_size: float, optional
        :param mut_rate: Scale of punctual mutations (how big the effect of 
                              mutation can be), defaults to 0.1
        :type mut_rate: float, optional
        :param n_parents: Number of parents contributing their genome
                          to a new individual, defaults to 2
        :type n_parents: int, optional
        :param allow_clones: If a code can occur as a parent multiple times when more 
                             than two parents are used, default to False.
        :type allow_clones: bool, optional
        :param temp: Temperature for controlling the softmax conversion
                     from scores to fitness (the actual prob. to sample 
                     a given parent for breeding), defaults to 1.
        :type temp: float, optional
        :param temp_factor: Multiplicative factor for temperature increase (`temp_factor` > 1)  
                            or decrease (0 < `temp_factor` < 1). Defaults to 1. indicating no change.
        :type temp: float, optional
        '''
        
        super().__init__(
            pop_size=pop_size,
            codes_shape=codes_shape,
            rnd_seed=rnd_seed,
            rnd_distr=rnd_distr,
            rnd_scale=rnd_scale
        )
        
        # Optimization hyperparameters
        self._mut_size     = mut_size
        self._mut_rate     = mut_rate
        self._n_parents    = n_parents
        self._allow_clones = allow_clones
        self._topk         = topk
        self._temp         = temp
        self._temp_factor  = temp_factor
    
    def __str__(self) -> str:
        ''' Return a string representation of the object for logging'''
        
        return f'GeneticOptimizer['\
                f'mut_size: {self._mut_size}'\
                f'mut_rate: {self._mut_rate}'\
                f'n_parents: {self._n_parents}'\
                f'allow_clones: {self._allow_clones}'\
                f'topk: {self._topk}'\
                f'temp: {self._temp}'\
                f'temp_factor: {self._temp_factor}'\
               ']'
    
    def __repr__(self) -> str: return str(self)
    
    def _step(
        self,
        data : Tuple[Score, ZdreamMessage],
        out_pop_size: int | None = None  
    ) -> Tuple[Codes, ZdreamMessage]:
        '''
        Optimizer step function that uses an associated score
        to each code to produce a new set of stimuli.

        :param data: Scores associated to each code and message.
        :type data: Tuple[Score, ZdreamMessage]
        :param out_pop_size: Population size for the next generation. 
                             Defaults to old one.
        :type out_pop_size: int | None, optional
        :return: Optimized set of codes and a message.
        :rtype: Tuple[Score, ZdreamMessage]
        '''
        
        super()._step(data=data)
        
        # Extract score and message
        scores, msg = data
        
        # Use old population size as default
        pop_size = default(out_pop_size, self.pop_size)      

        # Prepare data structure for the optimized codes
        codes_new = np.empty(shape=(pop_size, np.prod(self._codes_shape)), dtype=np.float32)

        # Get indices that would sort scores so that we can use it
        # to preserve the top-scoring stimuli
        sort_s = np.argsort(scores)
        topk_old_gen = self.codes[sort_s[-self._topk:]]
        
        # Convert scores to fitness (probability) via 
        # temperature-gated softmax function (needed only for rest of population)
        fitness = softmax(scores / self._temp)
        
        # The rest of the population is obtained by generating
        # children using breeding and mutation.
        
        # Breeding           
        new_gen = self._breed(
            population=self._codes.copy(),  # type: ignore
            pop_fitness=fitness,
            num_children=pop_size-self._topk,
        )
        
        # Mutating
        new_gen = self._mutate(
            population=new_gen
        )

        # New codes combining previous top-k codes and new generated ones
        codes_new[:self._topk] = topk_old_gen
        codes_new[self._topk:] = new_gen
        
        # Temperature 
        self._temp *= self._temp_factor
        
        # TODO: Add temperature rewarming of a factor `temp_rewarm()` if below treshold `temp_threshold`
        
        self._codes = codes_new.copy()
        
        return codes_new, msg

    def _breed(
        self,
        population  : NDArray,
        pop_fitness : NDArray,
        num_children : int | None = None
    ) -> NDArray:
        '''
        Perform breeding on the given population with given parameters.

        :param population: Population to breed using the fitness.
        :type population: NDArray.
        :param pop_fitness: Population fitness (i.e. probability to be selected
                            as parents for the next generation).
        :type pop_fitness: NDArray.
        :param num_children: Number of children in the new population, defaults to the
                             total number of codes (no parent preserved)
        :type num_children: int | None, optional
        
        :return: Breed population.
        :rtype: NDArray
        '''
        
        # NOTE: Overwrite old population

        # Number of children defaults to population size
        # i.e. no elements in the previous generation survives in the new one
        num_children = default(num_children, self.pop_size)
        
        families = self._rng.choice(
            a=self.pop_size,
            size=(num_children, self._n_parents),
            p=pop_fitness,
            replace=True
        ) if self._allow_clones and self._n_parents > 2 else np.stack([
            self._rng.choice(
                a = self.pop_size,
                size=self._n_parents,
                p=pop_fitness,
                replace=False,
            ) for _ in range(num_children)
        ])

        # Identify which parent contributes which genes for every child
        parentage = self._rng.choice(
            a=self._n_parents, 
            size=(num_children, *self._codes_shape), 
            replace=True
        )
        
        children = np.empty(shape=(num_children, np.prod(self._codes_shape)))

        for child, family, lineage in zip(children, families, parentage):
            for i, parent in enumerate(family):
                genes = lineage == i
                child[genes] = population[parent][genes]
                
        return children

    def _mutate(
        self,
        population : NDArray
    ) -> NDArray:
        '''
        Perform punctual mutation to given population using input parameters.

        :param population: Population of codes to mutate.
        :type population: NDArray
        :return: Mutated population.
        :rtype: NDArray
        '''

        # Compute mutation mask
        mut_loc = self._rng.choice(
            [True, False],
            size=population.shape,
            p=(self._mut_rate, 1 - self._mut_rate),
            replace=True
        )

        population[mut_loc] += self._rnd_sample(
            scale=self._mut_size, 
            size=mut_loc.sum()
        )
        
        return population 


class CMAESOptimizer(Optimizer):
    
    def __init__(
        self,
        pop_size    : int,
        codes_shape: int | Tuple[int, ...],
        rnd_seed   : None | int = None,
        rnd_distr  : RandomDistribution = 'normal',
        rnd_scale  : float = 1.,
        x0         : NDArray | None = None,
        sigma0     : float = 2,
    ) -> None:
        
        super().__init__(
            pop_size = pop_size, 
            codes_shape = codes_shape,
            rnd_seed = rnd_seed, 
            rnd_distr = rnd_distr,
            rnd_scale = rnd_scale
        )
        
        self._sigma0 = sigma0
        
        x0 = default(x0, np.zeros(shape=np.prod(self._codes_shape)))
        
        inopts = {'popsize': pop_size}
        if rnd_seed: inopts['seed'] = rnd_seed
        
        self._es = CMAEvolutionStrategy(
            x0     = x0,
            sigma0 = sigma0,
            inopts = inopts
        )
        
    def __str__ (self) -> str: return f'CMAESOptimizer[sigma0: {self._sigma0}]'
    def __repr__(self) -> str: return str(self)
    
    def _rnd_codes_generation(self, **kwargs) -> Codes:
        
        return np.stack(self._es.ask())

    
    def _step(self, data: Tuple[Score, ZdreamMessage]) -> Tuple[Codes, ZdreamMessage]:
        
        super()._step(data=data)
        
        scores, msg  = data
        
        self._es.tell(
            solutions=list(self._codes.copy()), # type: ignore
            function_values=list(-scores)
        )
        
        self._codes = np.stack(self._es.ask())
        
        return self._codes, msg