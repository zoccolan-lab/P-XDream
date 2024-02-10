'''
Collection of codes for testing the workings of zdream optimizers
'''

import unittest
import numpy as np
from numpy.typing import NDArray
from zdream.optimizer import GeneticOptimizer

from typing import Dict, cast

def non_convex_objective_fn(state : NDArray | Dict[str, NDArray]) -> NDArray:
    x, y = cast(NDArray, state)

    a1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
    a2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    score = a1 + a2 + 20
    return -score

class GeneticOptimizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.states_shape = (2,)
        self.random_state = 31415
        self.random_distr = 'normal'

        self.mutation_size = 0.3
        self.mutation_rate = 0.3
        self.population_size = 50
        self.temperature = 1.

        self.num_iteration = 100  
    
    def test_improvement_2_parents(self):
        optim = GeneticOptimizer(
            non_convex_objective_fn,
            states_shape=self.states_shape,
            random_state=self.random_state,
            random_distr=self.random_distr,
            mutation_rate=self.mutation_rate,
            mutation_size=self.mutation_size,
            population_size=self.population_size,
            temperature=self.temperature,
            num_parents=2,
        )

        # Initialize optimizer with random condition
        state = optim.init()

        print(optim._param)

        # Compute the score of the initial state
        init_score = optim.evaluate(state)

        for t in range(self.num_iteration):
            state = optim.step(state)

        # Extract optimizer states, in particular the
        # final score which is tested against the initial
        curr_score = cast(NDArray, optim.stats['curr_score'])

        print(optim.stats['best_shist'])

        self.assertGreater(curr_score.max(), init_score.max())

