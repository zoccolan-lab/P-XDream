import numpy as np
from abc import abstractmethod

from scipy.special import softmax

from .utils import SubjectScore

from .utils import Codes, SubjectState
from .utils import default, lazydefault
from .utils import Message

from typing import Callable, Dict, Tuple, List, cast
from numpy.typing import NDArray

ObjectiveFunction = Callable[[SubjectState], SubjectScore]

class Optimizer:
    '''
    Base class for generic optimizer, which keeps track of current
    parameter set (codes) and defines the abstract `step()` method 
    which every concrete instantiation should implement.
    '''
    
    def __init__(
        self,
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
        
        self._space = lazydefault(states_space, lambda : {i : (None, None) for i in range(len(states_shape))}) # type: ignore
        self._shape = lazydefault(states_shape, lambda : (len(states_space),))                                 # type: ignore
        
        self._param     : List[NDArray] = []
        self._score     : List[NDArray] = []
        self._score_nat : List[NDArray] = []
        self._distr = random_distr
        
        # Initialize the internal random number generator for reproducibility
        self._rng = np.random.default_rng(random_state)
        
    @property
    def n_states(self) -> int:
        return 1
    
        
    def init(self, init_cond : str | NDArray = 'normal', **kwargs) -> Codes:
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
            self._param = [init_cond.copy()]
        else:
            self._distr = init_cond
            self._param = [self.rnd_sample(size=(self.n_states, *self._shape), **kwargs)]

        return self.param
    
    @abstractmethod
    def step(self, states : SubjectScore) -> Codes:
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
    
    def _get_stats(self, score: List[NDArray], param: bool = False):
        #TO DO: i just quickfixed the issue when nr generated imgs != nr nat imgs
        #we can probably find  a more elegant solution
        
        flat_idx : np.intp    = np.argmax(score)
        hist_idx : List[int]  = np.argmax(score, axis=1)

        best_gen, *best_idx = np.unravel_index(flat_idx, np.shape(score))
        
        return {
            'best_score' : score[best_gen][best_idx],
            'best_param' : self._param[best_gen][best_idx] if param else None,
            'curr_score' : score[-1],
            'curr_param' : self._param[-1] if param else None,
            'mean_shist' : np.array([np.mean(s) for s in score]),
            'best_shist' : [score[idx] for score, idx in zip(score, hist_idx)],
            'best_phist' : [param[idx] for param, idx in zip(self._param, hist_idx)]
                if param else None,
        }
        
    @property
    def stats(self):
        return self._get_stats(score=self._score, param = True)
    
    @property
    def stats_nat(self):
        return self._get_stats(score=self._score_nat)
        
        
    @property
    def solution(self) -> NDArray:
        flat_idx : np.intp    = np.argmax(self._score)
        best_gen, *best_idx = np.unravel_index(flat_idx, np.shape(self._score))
        
        return self._param[best_gen][best_idx]
        
    @property
    def score(self) -> NDArray:
        return self._score[-1]
    
    @property
    def param(self) -> NDArray:
        return self._param[-1] if self._param else np.array([])
    
class GeneticOptimizer(Optimizer):
    '''
    Optimizer that implements a genetic optimization strategy.
    In particular these optimizer devise a population of candidate
    solutions (set of parameters) and iteratively improves the
    given objective function via the following heuristics:
    - The top_k performing solution are left unaltered
    - The rest of the population pool are recombined to produce novel
      candidate solutions via breeding and random mutations
    - The num_parents contributing to a single offspring are selected
      via importance sampling based on parents fitness scores
    - Mutations rate and sizes can be adjusted independently 
    '''
    
    def __init__(
        self,
        states_space : None | Dict[int | str, Tuple[float | None, float | None]] = None,
        states_shape : None | int | Tuple[int, ...] = None,
        random_state : None | int = None,
        random_distr : None | str = None,
        mutation_size : float = 0.1,
        mutation_rate : float = 0.3,
        population_size : int = 50,
        temperature : float = 1.,
        num_parents : int = 2,
    ) -> None:
        '''
        :param objective_fn: Objective function used to convert
            observables (states) to scores
        :param states_space: Currently NOT USED
        :param states_shape: Tuple defining the shape of the
            optimization space (assumed free of constraints)
        :param random_state: Seed for random number generation
        :param random_distr: Name of distribution to use to sample
            initial conditions if not directly provided
        :param mutation_size: Scale of punctual mutations (how big
            the effect of mutation can be)
        :param mutation_rate: Probability of single-point mutation
        :param population_size: Number of subject in the population
        :param temperature: Temperature for controlling the softmax
            conversion from scores to fitness (the actual prob. to
            sample a given parent for breeding)
        :param num_parents: Number of parents contributing their
            genome to a new individual
        '''
        
        super().__init__(
            states_space,
            states_shape,
            random_state, 
            random_distr,
        )
        
        self.num_parents = num_parents
        self.temperature = temperature
        self.mutation_size = mutation_size
        self.mutation_rate = mutation_rate
        self.init_pop_size = population_size

    @property
    def n_states(self) -> int:
        return len(self.param) if self._param else self.init_pop_size
    
    def step(
        self,
        data : Tuple[SubjectScore, Message],
        out_size: int | None = None,
        temperature : float | None = None, 
        save_topk : int = 2,   
    ) -> Codes:
        '''
        Optimizer step function where current observable (states)
        are scored using the internal objective function and a
        novel set of parameter is proposed that would hopefully
        increase future scores.

        # TODO: Update doc
        :param data: Set of computed scores (gather from a
            scorer that evaluated for example some ANN activations).
        :param out_size: TODO
        :param temperature: Temperature in the softmax conversion
            from scores to fitness, i.e. the actual probabilities of
            selecting a given subject for reproduction
        :type temperature: positive float (> 0)
        :param save_topk: Number of top-performing subject to preserve
            unaltered during the current generation (to avoid loss
            of provably decent solutions)
        :type save_topk: positive int (> 0)

        :returns new_pop: Novel set of parameters (new population pool)
        :rtype: Numpy array
        '''
        
        
        # Use Message mask to filter for generated data
        curr_scores, msg = data
        nat_scores  =  curr_scores[~msg.mask]
        curr_scores = curr_scores[msg.mask]
        
        pop_size    = default(out_size, self.n_states)

        temperature = default(temperature, self.temperature)

        # Prepare new parameter (population) set
        new_param = np.empty(shape=(pop_size, *self._shape))

        # Get indices that would sort scores so that we can use it
        # to preserve the top-scoring subject
        sort_s = np.argsort(curr_scores)
        topk_p = self.param[sort_s[-save_topk:]]
        # rest_p = self.param[sort_idx[:-save_topk]]
        # rest_s = self.score[sort_idx[:-save_topk]]

        # Convert scores to fitness (probability) via temperature-
        # gated softmax function (needed only for rest of population)
        # fitness = softmax(rest_s / temperature)
        fitness = softmax(curr_scores / temperature)
        
        new_param[:save_topk] = topk_p

        # The rest of the population is obtained by generating
        # children (breed) and mutating them
        next_gen = self._breed(
            population=self.param,
            pop_fitness=fitness,
            num_children=pop_size - save_topk,
        )

        new_param[save_topk:] = self._mutate(
            population=next_gen,
        )

        # Bookkeeping: update internal parameter history
        # and overall score history
        # NOTE: These two lists are NOT aligned. They are
        #       off-by-one as observed states correspond
        #       to last parameter set and we are devising
        #       the new one right now (hence why old_score)
        self._score.append(curr_scores)
        self._score_nat.append(nat_scores)
        self._param.append(new_param)
        

        return new_param
    
    def _mutate(
        self,
        mut_rate : float | None = None,
        mut_size : float | None = None,
        population : NDArray | None = None,
    ) -> NDArray:
        mut_rate = default(mut_rate, self.mutation_rate)
        mut_size = default(mut_size, self.mutation_size)
        population = default(population, self.param)

        # Identify mutations spots for every subject in the population
        mutants = population.copy()
        mut_loc = self._rng.choice([True, False],
            size=mutants.shape,
            p=(mut_rate, 1 - mut_rate),
            replace=True
        )

        mutants[mut_loc] += self.rnd_sample(scale=mut_size, size=mut_loc.sum())

        return mutants 
    
    def _breed(
        self,
        population : NDArray | None = None,
        pop_fitness : NDArray | None = None,
        num_children : int | None = None,
        allow_clones : bool = False,
    ) -> NDArray:
        # NOTE: We use lazydefault here because .param and .score might
        #       not be populated (i.e. first call to breed) but we don't
        #       want to fail if either population or fitness are provided
        population  = lazydefault(population,  lambda : self.param)
        pop_fitness = lazydefault(pop_fitness, lambda : self.score)
        num_children = default(num_children, len(population))
        
        # Select the breeding family based on the fitness of each parent
        # NOTE: The same parent can occur more than once in each family
        families = self._rng.choice(
            len(population),
            size=(num_children, self.num_parents),
            p=pop_fitness,
            replace=True,
        ) if allow_clones else np.stack([
            self._rng.choice(
                len(population),
                size=self.num_parents,
                p=pop_fitness,
                replace=False,
            ) for _ in range(num_children)
        ])

        # Identify which parent contributes which genes for every child
        # NOTE: First dimension of self._shape is the total population size
        parentage = self._rng.choice(self.num_parents, size=(num_children, *self._shape), replace=True)
        children = np.empty(shape=(num_children, *self._shape))

        for c, (child, family, lineage) in enumerate(zip(children, families, parentage)):
            for parent in family:
                genes = lineage == parent
                child[genes] = population[parent][genes]
                
        return children