import numpy as np
from abc import abstractmethod

from typing import Callable

class Optimizer:
    '''
    Base class for generic optimizer, which keeps track of current
    parameter state (codes) and 
    - Implements the `step()` method which
        - Takes the scores corresponding to the images
        - Updates the samples to increase expected scores
    '''
    
    def __init__(self, objective_fn : Callable) -> None:
        pass
    
    @abstractmethod
    def step(self, activations) -> np.ndarray:
        '''
        '''
        raise NotImplementedError('Optimizer is abstract. Use concrete implementations')
    
class GeneticOptimizer(Optimizer):
    '''
    '''
    
    def step(self) -> np.ndarray:
        pass