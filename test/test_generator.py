'''
Collection of codes for testing the workings of zdream generators
'''

import unittest
import torch
from torch import Tensor
from zdream.generator import InverseAlexGenerator
from zdream.utils import SubjectState

class InverseAlexGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        
        # NOTE: One must change this root dir accordingly
        #       for proper testing to work
        self.root = '/media/pmurator/archive/InverseAlexGenerator'
        
        self.batch = 2
        self.target_shape = (3, 256, 256)
        
    def run_mock_inp(self, generator : InverseAlexGenerator) -> Tensor:
        # Test generator on mock input
        inp_dim = generator.input_dim
        mock_inp = torch.randn(self.batch, *inp_dim, device=generator.device)
        
        return generator(mock_inp)
    
    def test_loading_fc8(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='fc8'
        ).to('cuda')
        
        out = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.batch, *self.target_shape))
    
    def test_loading_fc7(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='fc7'
        ).to('cuda')
        
        out = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.batch, *self.target_shape))

    def test_loading_conv(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='conv4'
        ).to('cuda')

        out = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.batch, *self.target_shape))
    
    def test_loading_norm1(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='norm1'
        ).to('cuda')
        
        out = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.batch, *self.target_shape))

    
    def test_loading_norm2(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='norm2'
        ).to('cuda')
        
        out = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.batch, *self.target_shape))
    
    def test_loading_pool(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='pool5'
        ).to('cuda')
        
        out = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.batch, *self.target_shape))
