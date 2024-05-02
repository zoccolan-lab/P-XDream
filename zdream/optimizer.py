'''
This file contains the implementation of the `Optimizer` class and its subclasses.
It provides a set of classes that implement different optimization strategies:
- `GeneticOptimizer`: Optimizer that implements a genetic optimization strategy.
- `CMAESOptimizer`: Optimizer that implements a Covariance Matrix Adaptation Evolution Strategy.
'''

from abc import ABC, abstractmethod
from typing import Callable, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax
from cma import CMAEvolutionStrategy

from .utils.types import Codes, Scores
from .utils.misc import default

RandomDistribution = Literal['normal', 'gumbel', 'laplace', 'logistic']
''' 
Name of distributions for random codes initializations
'''

class Optimizer(ABC):
    '''
    Abstract class for a generic optimizer intended to maximize an objective.
    It implements a function `step()` to produce a new set of codes based on a scoring input.
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
        Initialize a new gradient free optimizer with proper population size and codes shape.

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
        It raises an error in the case no codes are available.
        
        Codes are internally linearized, the property
        handles codes reshaping to the expected to the expected shape.
        
        :return: Last produced codes.
        :rtype: Codes
        '''
        
        # Codes not available check
        if self._codes is None:
            err_msg = 'No codes available. Use `init()` method to generate the first codes.'
            raise ValueError(err_msg)
        
        # Extract population size
        pop_size, *_ = self._codes.shape
        
        # Reshape codes to the expected shape
        codes_ = np.reshape(self._codes, (pop_size, *self._codes_shape))

        return codes_.copy()
    
    
    @property
    def pop_size(self) -> int:
        ''' 
        Number of codes the optimizer is optimizing for.
        NOTE:   The number of codes can change dynamically 
                during the optimization process
        '''
        
        try:
            pop_size, *_ = self.codes.shape
            return pop_size
        
        except ValueError:
            return self._init_n_codes
    
    # --- STEP ---
    
    def step(self, scores: Scores) -> Codes:
        '''
        Wrapper for actual step implementation in `_step()` that automatizes 
        saving of last generation codes in `self_codes`.
        
        :param scores: Tuple containing the score associated to each old code.
        :type scores: Score
        :return: Set of new codes supposed to produce an higher value of the objective.
        :rtype: Codes
        ''' 
        
        self._codes = self._step(scores=scores)
        
        return self.codes
    
    
    @abstractmethod
    def _step(self, scores: Scores) -> Codes:
        '''
        Abstract step method.
        The `step()` method receives the scores associated to the
        last produced codes and uses them to produce a new set of codes.
        
        By defaults it only checks if codes are available.
        
        :param scores: Tuple containing the score associated to each old code.
        :type scores: Scores
        :return: Set of new codes to be used to improve future states scores.
        :rtype: Codes
        '''        
        
        if self._codes is None:
            err_msg = 'No codes provided, use `init()` to generate first ones. '
            raise ValueError(err_msg)
        
        pass
    
    
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
                err_msg =   f'Provided initial codes have shape: {init_codes.shape}, '\
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
        '''
        Generate random codes using the specified distribution.
        It uses additional parameters passed as kwargs to the random generator.
        
        :return: Randomly generated codes.
        :rtype: Codes
        '''
        
        return self._rnd_sample(
            size=(self._init_n_codes, np.prod(self._codes_shape)),
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
        
        # TODO Parameter domain sanity check
        
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
        
    # --- STRING REPRESENTATION ---
    
    def __str__(self) -> str:
        ''' Return a string representation of the object for logging'''
        
        return  f'GeneticOptimizer['\
                f'mut_size: {self._mut_size}'\
                f'mut_rate: {self._mut_rate}'\
                f'n_parents: {self._n_parents}'\
                f'allow_clones: {self._allow_clones}'\
                f'topk: {self._topk}'\
                f'temp: {self._temp}'\
                f'temp_factor: {self._temp_factor}'\
                ']'
    
    def __repr__(self) -> str: return str(self)
    ''' Return a string representation of the object''' 
    
    def _step(
        self,
        scores : Scores,
        out_pop_size : int | None = None  
    ) -> Codes:
        '''
        Optimizer step function that uses an associated score
        to each code to produce a new set of stimuli.

        :param scores: Scores associated to each code.
        :type scores: Score
        :param out_pop_size: Population size for the next generation. 
            Defaults to old one.
        :type out_pop_size: int | None, optional
        :return: Optimized set of codes.
        :rtype: Score
        '''
        
        super()._step(scores=scores)
        
        # Use old population size as default
        pop_size = default(out_pop_size, self.pop_size)      

        # Prepare data structure for the optimized codes
        codes_new = np.empty(shape=(pop_size, np.prod(self._codes_shape)), dtype=np.float32)

        # Get indices that would sort scores so that we can use it
        # to preserve the top-scoring stimuli
        sort_s = np.argsort(scores)
        topk_old_gen = self._codes[sort_s[-self._topk:]]
        
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
        
        return codes_new

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
        
        # We use clones if specified and if the number of parents is greater than 2
        use_clones = self._allow_clones and self._n_parents > 2
        
        # Select parents for each child with the two strategies
        
        # In the case of clones we do sampling with replacement
        families = self._rng.choice(
            a=self.pop_size,
            size=(num_children, self._n_parents),
            p=pop_fitness,
            replace=True 
        # Otherwise we sample one element at a time without replacement
        # and combine them to form the family
        ) if use_clones else np.stack([
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
            size=(num_children, np.prod(self._codes_shape)), 
            replace=True
        )
        
        # Generate empty children
        children = np.empty(shape=(num_children, np.prod(self._codes_shape)))

        # Fill children with genes from selected parents
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

        # Apply mutation
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
        sigma0     : float = 1.,
    ) -> None:
        '''
        Initialize a new CMAESOptimizer with initial Multivariate gaussian
        mean vector and variance for the covariance matrix as initial parameters.

        :param pop_size: Number of initial codes.
        :type pop_size: int
        :param codes_shape: Codes shape. If one dimensional the single dimension supported.
        :type codes_shape: int | Tuple[int, ...]
        :param rnd_seed: Random state for pseudo-random numbers generation.
        :type rnd_seed: None | int, optional
        :param rnd_distr: Nature of the random distribution for initial
        :type rnd_distr: RandomDistribution, optional
        :param rnd_scale: Scale for initial codes generation, defaults to 1.
        :type rnd_scale: float, optional
        :param x0: Initial mean vector for the multivariate gaussian distribution, defaults to None that 
            is a zero mean vector.
        :type x0: NDArray | None, optional
        :param sigma0: Initial variance for the covariance matrix, defaults to 1.
        :type sigma0: float, optional
        '''
        
        super().__init__(
            pop_size=pop_size, 
            codes_shape=codes_shape,
            rnd_seed=rnd_seed, 
            rnd_distr=rnd_distr,
            rnd_scale=rnd_scale
        )
        
        # Save variance for the covariance matrix
        self._sigma0 = sigma0
        
        # Use zero mean vector if not provided
        x0 = default(x0, np.zeros(shape=np.prod(self._codes_shape)))
        
        # Create dictionary for CMA-ES settings
        inopts = {'popsize': pop_size}
        if rnd_seed: inopts['seed'] = rnd_seed
        
        # Initialize CMA-ES optimizer
        self._es = CMAEvolutionStrategy(
            x0     = x0,
            sigma0 = sigma0,
            inopts = inopts
        )
    
    # --- STRING REPRESENTATION ---
    def __str__ (self) -> str: return f'CMAESOptimizer[sigma0: {self._sigma0}]'
    ''' Return a string representation of the object '''
    
    def __repr__(self) -> str: return str(self)
    ''' Return a string representation of the object '''


    # --- STEP ---

    def _step(self, scores: Scores) -> Codes:
        '''
        Perform a step of the optimization process using the CMA-ES optimizer.

        :param scores: Tuple containing the score associated to each old code.
        :type scores: Scores
        :return: New set of codes.
        :rtype: Codes
        '''
        
        super()._step(scores=scores)
        
        self._es.tell(
            solutions=list(self._codes.copy()), # type: ignore
            function_values=list(-scores)
        )
        
        self._codes = np.stack(self._es.ask())
        
        return self._codes
    
    def _rnd_codes_generation(self, **kwargs) -> Codes:
        '''
        Override super method to generate random codes using the CMA-ES optimizer.

        :return: Randomly generated codes using current CMA-ES optimizer state.
        :rtype: Codes
        '''
        
        return np.stack(self._es.ask())