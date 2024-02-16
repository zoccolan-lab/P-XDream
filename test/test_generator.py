'''
Collection of codes for testing the workings of zdream generators
'''

import unittest
import torch
import numpy as np
from typing import Tuple
from zdream.generator import InverseAlexGenerator
from zdream.utils import Stimuli, Message
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from zdream.utils import device

class _RandomImageDataset(Dataset):
    
    def __init__(self, num_images: int, image_size: Tuple[int, ...]):
        self.num_images = num_images
        self.image_size = image_size
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, _) -> Tensor:
                
        return torch.tensor(np.random.rand(*self.image_size), dtype=torch.float32)

class InverseAlexGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        
        # NOTE: One must change this root dir accordingly
        #       for proper testing to work
        self.root = '/media/pmurator/archive/InverseAlexGenerator'
        self.root = 'C://Users//user.LAPTOP-G27BJ7JO//Documents//GitHub//ZXDREAM//data//InverseAlexGenerator'
        
        self.gen_batch = 2
        self.target_shape = (3, 256, 256)
        
    def run_mock_inp(self, generator : InverseAlexGenerator) -> Tuple[Stimuli, Message]:
        # Test generator on mock input
        inp_dim = generator.input_dim
        mock_inp = torch.randn(self.gen_batch, *inp_dim, device=generator.device)
        
        return generator(mock_inp)
        
    def _get_data_loader(self, num_images: int, batch_size: int, target_shape: Tuple[int, ...]):
        
        dataset = _RandomImageDataset(
            num_images=num_images,
            image_size=target_shape
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size
        )
        
        return dataloader
    
    def test_loading_fc8(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='fc8'
        ).to(device)
        
        out, msg = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.gen_batch, *self.target_shape))
        self.assertTrue(all(msg.mask))
    
    def test_loading_fc7(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='fc7'
        ).to(device)
        
        out, msg = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.gen_batch, *self.target_shape))
        self.assertTrue(all(msg.mask))

    def test_loading_conv(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='conv4'
        ).to(device)

        out, msg = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.gen_batch, *self.target_shape))
        self.assertTrue(all(msg.mask))

    def test_loading_norm1(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='norm1'
        ).to(device)
        
        out, msg = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.gen_batch, *self.target_shape))
        self.assertTrue(all(msg.mask))

    
    def test_loading_norm2(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='norm2'
        ).to(device)
        
        out, msg = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.gen_batch, *self.target_shape))
        self.assertTrue(all(msg.mask))
    
    def test_loading_pool(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='pool5'
        ).to(device)
        
        out, msg = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.gen_batch, *self.target_shape))
        self.assertTrue(all(msg.mask))
        
    def test_gen_plus_nat_batch(self):
        
        num_images = 5
        num_nat    = 18
        dataloader_bach = 1
                
        dataloader = self._get_data_loader(
            num_images=num_images,
            batch_size=dataloader_bach,
            target_shape=self.target_shape
        )
        
        generator = InverseAlexGenerator(
            root=self.root,
            variant='fc8',
            mixing_mask=[True] * self.gen_batch + [False] * num_nat,
            data_loader=dataloader
        ).to(device)

        out, msg = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.gen_batch+num_nat, *self.target_shape))
        
        self.assertTrue(np.sum( msg.mask) == self.gen_batch)
        self.assertTrue(np.sum(~msg.mask) == num_nat)
        
    
