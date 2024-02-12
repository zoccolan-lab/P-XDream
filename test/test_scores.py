'''
Collection of codes for testing the workings of zdream scores
'''

import unittest
import numpy as np
from numpy.typing import NDArray
from zdream.scores import MinMaxSimilarity

from typing import Dict, cast

class MinMaxSimilarityTest(unittest.TestCase):
    
    random_state = 31415
    
    def setUp(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
    
    def test_cnn_two_layers_defaults(self):
        # Initialize scorer and create mock up data to test
        score = MinMaxSimilarity(
            positive_target='conv8',
            negative_target='conv1',
            neg_penalty=1.5,
            similarity_fn=None,
            grouping_fn=None,
        )
        
        num_imgs = 5
        num_unit_conv1 = 500
        num_unit_conv8 = 250
        
        state_1 = {
            'conv1' : np.random.uniform(-1, +1, size=(2 * num_imgs, num_unit_conv1)),
            'conv8' : np.random.uniform(-1, +1, size=(2 * num_imgs, num_unit_conv8)),
        }
        
        state_2 = {
            'conv1' : np.ones((2 * num_imgs, num_unit_conv1)), # All equal => zero neg. penalty
            'conv8' : np.random.uniform(-1, +1, size=(2 * num_imgs, num_unit_conv8))
        }
        
        # Score these states via scoring function. We should get a score for
        # each pair of images, i.e. expected output has shape (num_images,)
        ans_1 = score(state_1)
        ans_2 = score(state_2)

        self.assertEqual(ans_1.shape, (num_imgs,))
        self.assertEqual(ans_2.shape, (num_imgs,))
        
        self.assertTrue(np.all(ans_2 > 0))
