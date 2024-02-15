'''
Collection of codes for testing the workings of zdream scores
'''

from typing import Dict, cast
import unittest
import numpy as np
from numpy.typing import NDArray
from zdream.scores import MSEScore, MinMaxSimilarity

class MSEScoreTest(unittest.TestCase):
    
    rseed = 123 
    batch = 5  
    nkeys = 3
    
    def setUp(self) -> None:
        np.random.seed(self.rseed) 
        
    def test_state_array(self):
        
        target = np.random.rand(self.batch, 3, 224, 224)
        state  = np.random.rand(self.batch, 3, 224, 224)
        mse_score = MSEScore(target=target)
        score = cast(NDArray, mse_score(state=state))
        
        # Check if it's a one-dimensional array of float32
        # with the same length as batch size
        self.assertTrue(score.ndim == 1)
        self.assertEqual(len(score), self.batch)
        self.assertTrue(score.dtype == np.float32)
        
        # Check if all values are non-positive
        self.assertTrue(np.all(score <= 0))
        
    def test_target_array(self):
        
        target = np.random.rand(self.batch, 3, 224, 224)
        mse_score = MSEScore(target=target)
        score = cast(NDArray, mse_score(state=target))
        
        # Check target score to be zero
        self.assertTrue(np.all(score == 0))
        
    def test_state_dict(self):
        
        target = {
            f'key-{i}': np.random.rand(self.batch, 3, 224, 224)
            for i in range(self.nkeys)
        }
        state = {
            f'key-{i}': np.random.rand(self.batch, 3, 224, 224)
            for i in range(self.nkeys)
        }
        mse_score = MSEScore(target=target)
        score = cast(Dict[str, NDArray], mse_score(state=state))
        
        # Check if the score and state dictionaries have the keys
        self.assertEqual(state.keys(), score.keys())
        
        # Check if any arrays satisfy requirements of being a one
        # dimensional float.3d array of the same size as the batch
        # containing non positive values
        
        for arr in score.values():
            
            self.assertTrue(arr.ndim == 1)
            self.assertEqual(len(arr), self.batch)
            self.assertTrue(arr.dtype == np.float32)
            self.assertTrue(np.all(arr <= 0))
            
        # Check if target can have more key then the subject state has
        target["new_key"] = np.random.rand(self.batch, 3, 224, 224)
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
        score = cast(Dict[str, NDArray], mse_score(state=target))
        
        # Check all targets score to be zero
        for arr in score.values():
            self.assertTrue(np.all(arr == 0))
        

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
