'''
Collection of codes for testing the workings of zdream scores
'''

import unittest
import numpy as np

from itertools import combinations

from zdream.utils import Message, SubjectState
from zdream.scores import MSEScorer, WeightedPairSimilarityScorer, MaxActivityScorer

class MSEScorerTest(unittest.TestCase):
    
    rseed = 123 
    batch = 5  
    nkeys = 3
    
    def setUp(self) -> None:
        np.random.seed(self.rseed)
        
        self.msg = Message(mask=np.ones(self.batch, dtype=bool))
        
    def test_score(self):
        
        # Define random target and state        
        target = {
            f'key-{i}': np.random.rand(self.batch, 3, 224, 224)
            for i in range(self.nkeys)
        }
        state = {
            f'key-{i}': np.random.rand(self.batch, 3, 224, 224)
            for i in range(self.nkeys)
        }
        
        mse_score = MSEScorer(target=target)
        score, _ = mse_score(data=(state, self.msg))
        
        # Check if it's a one-dimensional array of float32
        # with the same length as batch size
        self.assertTrue(score.ndim == 1)
        self.assertEqual(len(score), self.batch)
        self.assertTrue(score.dtype == np.float32)
        
        # Check if all values are non-positive
        self.assertTrue(np.all(score <= 0))
        
    def test_state_dict(self):
        
        # Define random target and state  
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
        
        mse_score = MSEScorer(target=target)
        
        score, _ = mse_score(data=(state, self.msg))
        self.assertIsNotNone(score)
        
        # Check if subject state raises errors when it has unexpected
        # keys for the target
        state["new_key2"] = np.random.rand(self.batch, 3, 224, 224)
        
        with self.assertRaises(ValueError):
            mse_score(data=(state, self.msg))
        
    def test_target_dict(self):
        
        target = {
            f'key-{i}': np.random.rand(self.batch, 3, 224, 224)
            for i in range(self.nkeys)
        }
        mse_score = MSEScorer(target=target)
        score, _ = mse_score(data=(target, self.msg))
        
        # Check all targets score to be zero
        self.assertTrue(np.allclose(score, 0))
        
class WeightedPairSimilarityScorerTest(unittest.TestCase):
    
    num_imgs = 5
    random_state = 31415
    
    def setUp(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        self.msg = Message(mask=np.ones(self.num_imgs, dtype=bool))
        
    def test_two_layers_euclidean(self):
        signature = {
            'conv1' : +1.,
            'conv8' : -1.,
        }

        # Initialize scorer and create mock up data to test
        score = WeightedPairSimilarityScorer(
            signature=signature,
            metric='euclidean',
            filter_distance_fn=None,
        )
        
        num_unit_conv1 = 500
        num_unit_conv8 = 250
        
        state_1 = {
            'conv1' : np.random.uniform(-1, +1, size=(self.num_imgs, num_unit_conv1)),
            'conv8' : np.random.uniform(-1, +1, size=(self.num_imgs, num_unit_conv8)),
        }
        
        state_2 = {
            'conv1' : np.ones((self.num_imgs, num_unit_conv1)),
            'conv8' : np.random.uniform(-1, +1, size=(self.num_imgs, num_unit_conv8)),
        }
        
        # Scorer these states via scoring function. We should get a score for
        # each pair of images, i.e. expected output has shape (num_images,)
        score_1, _ = score(data=(state_1, self.msg))
        score_2, _ = score(data=(state_2, self.msg))

        expected_num_score = len(list(combinations(range(self.num_imgs), 2)))
        self.assertEqual(score_1.shape, (expected_num_score,))
        self.assertEqual(score_2.shape, (expected_num_score,))
        
        # Second scores has ensured similarity in late conv8 so that  
        self.assertTrue(np.all(score_2 > score_1))
        
class MaxActivityScorerTest(unittest.TestCase):
    
    num_imgs = 5
    random_state = 31415
    
    def setUp(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        self.msg = Message(mask=np.ones(self.num_imgs, dtype=bool))
        
    def test_score_format(self):
        
        scorer = MaxActivityScorer(
            neurons={'one': [4, 100], 'two': [45, 78], 'three': [1]},
            aggregate=lambda x: np.sum(np.stack(list(x.values())), axis=0).astype(np.float32)
        )
        
        mock_activations: SubjectState = {
            'one': np.random.randn(self.num_imgs, 200),
            'two': np.random.randn(self.num_imgs, 200),
            'three': np.random.randn(self.num_imgs, 200),
        }
        
        score, _ = scorer(data=(mock_activations, self.msg))
        
        self.assertEqual(score.ndim, 1)
        self.assertEqual(score.dtype, np.float32)
        self.assertEqual(len(score), self.num_imgs)