'''
Collection of codes for testing the workings of zdream scores
'''

from typing import Dict, cast
import unittest
import numpy as np
from numpy.typing import NDArray
from zdream.scores import MSEScore, WeightedPairSimilarity

class MSEScoreTest(unittest.TestCase):
    
    rseed = 123 
    batch = 5  
    nkeys = 3
    
    def setUp(self) -> None:
        np.random.seed(self.rseed) 
        
    def test_state_array(self):
        
        target = {
            f'key-{i}': np.random.rand(self.batch, 3, 224, 224)
            for i in range(self.nkeys)
        }
        state = {
            f'key-{i}': np.random.rand(self.batch, 3, 224, 224)
            for i in range(self.nkeys)
        }
        mse_score = MSEScore(target=target)
        score = cast(NDArray, mse_score(state=state))
        
        # Check if it's a one-dimensional array of float32
        # with the same length as batch size
        self.assertTrue(score.ndim == 1)
        self.assertEqual(len(score), self.batch)
        self.assertTrue(score.dtype == np.float32)
        
        # Check if all values are non-positive
        self.assertTrue(np.all(score <= 0))
        
    def test_state_dict(self):
        
        target = {
            f'key-{i}': np.random.rand(self.batch, 3, 224, 224)
            for i in range(self.nkeys)
        }
        state = {
            f'key-{i}': np.random.rand(self.batch, 3, 224, 224)
            for i in range(self.nkeys)
        }
        
        # Check if target can have more key then the subject state has
        target["new_key"] = np.random.rand(self.batch, 3, 224, 224)
        
        mse_score = MSEScore(target=target)
        self.assertIsNotNone(mse_score(state=state))
        
        # Check if subject state raises errors when it has unexpected
        # keys for the target
        state["new_key2"] = np.random.rand(self.batch, 3, 224, 224)
        with self.assertRaises(AssertionError):
            mse_score(state=state)
        
    def test_target_dict(self):
        
        target = {
            f'key-{i}': np.random.rand(self.batch, 3, 224, 224)
            for i in range(self.nkeys)
        }
        mse_score = MSEScore(target=target)
        score = mse_score(state=target)
        
        # Check all targets score to be zero
        self.assertTrue(np.allclose(score, 0))
        
class WeightedPairSimilarityTest(unittest.TestCase):
    
    random_state = 31415
    
    def setUp(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
    
    def test_two_layers_euclidean(self):
        signature = {
            'conv1' : +1.,
            'conv8' : -1.,
        }

        # Initialize scorer and create mock up data to test
        score = WeightedPairSimilarity(
            signature=signature,
            metric='euclidean',
            pair_fn=None
        )
        
        num_imgs = 5
        num_unit_conv1 = 500
        num_unit_conv8 = 250
        
        state_1 = {
            'conv1' : np.random.uniform(-1, +1, size=(2 * num_imgs, num_unit_conv1)),
            'conv8' : np.random.uniform(-1, +1, size=(2 * num_imgs, num_unit_conv8)),
        }
        
        state_2 = {
            'conv1' : np.ones((2 * num_imgs, num_unit_conv1)),
            'conv8' : np.random.uniform(-1, +1, size=(2 * num_imgs, num_unit_conv8)),
        }
        
        # Score these states via scoring function. We should get a score for
        # each pair of images, i.e. expected output has shape (num_images,)
        score_1 = score(state_1)
        score_2 = score(state_2)

        self.assertEqual(score_1.shape, (num_imgs,))
        self.assertEqual(score_2.shape, (num_imgs,))
        
        # Second scores has ensured similarity in late conv8 so that  
        self.assertTrue(np.all(score_2 > score_1))
