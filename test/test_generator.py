'''
Collection of codes for testing the workings of zdream generators
'''

from os import path
import random
import unittest
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from zdream.generator import InverseAlexGenerator
from zdream.utils import Stimuli, Message
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from zdream.utils import device, read_json

test_folder = path.dirname(path.abspath(__file__))
test_settings_fp = path.join(test_folder, 'local_settings.json')
test_settings: Dict[str, Any] = read_json(path=test_settings_fp)

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
        
        random.seed = 123
        
        # NOTE: One must edit the test_local_settings.json in test dir
        #       accordingly to proper local machine path
        self.root = test_settings['inverse_alex_net']
        
        self.gen_batch = 2
        self.target_shape = (3, 256, 256)
        
    def run_mock_inp(self, generator : InverseAlexGenerator) -> Tuple[Stimuli, Message]:
        # Test generator on mock input
        inp_dim = generator.input_dim
        mock_inp = torch.randn(self.gen_batch, *inp_dim, device=generator.device)
        
        return generator(mock_inp)
    
    def run_mock_inp_with_nat(self, generator : InverseAlexGenerator, mask: List[bool]) -> Tuple[Stimuli, Message]:
        # Test generator on mock input
        inp_dim = generator.input_dim
        mock_inp = torch.randn(self.gen_batch, *inp_dim, device=generator.device)
        
        return generator(mock_inp, mask)
    
        
    def _get_generator_nat(
        self,
        num_images: int, 
        batch_size: int,
        target_shape: Tuple[int, ...]
    ) -> InverseAlexGenerator:
        
        dataset = _RandomImageDataset(
            num_images=num_images,
            image_size=target_shape
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size
        )
        
        generator = InverseAlexGenerator(
            root=self.root,
            variant='fc8',
            data_loader=dataloader
        ).to(device)
        
        return generator
    
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
        
    def test_natural_images(self):
        
        # We run different cases for the number of images in the dataset,
        # the number of natural images in the generator and the number
        # of batches of the dataloader
        
        # num_images, num_nat, dataloader_bach
        parameters = [
            (100, 10,  1), # single batch
            (100, 10,  2), # batch divisible for the natural images
            (100, 10,  3), # batch non divisible for the natural images
            ( 71, 15, 12),
            ( 10, 10, 10), # same batch size as dataset
            ( 10, 20, 10), # more natural images than ones in the dataset
            ( 10,  0, 10), # no natural images
        ]
        
        for num_images, num_nat, dataloader_bach in parameters:
                
            generator = self._get_generator_nat(
                num_images=num_images,
                batch_size=dataloader_bach,
                target_shape=self.target_shape
            )
            
            mask = [True]*self.gen_batch + [False]*num_nat
            random.shuffle(mask)
            
            out, msg = self.run_mock_inp_with_nat(
                generator=generator,
                mask=mask
            )
        
            self.assertEqual(out.shape, (self.gen_batch+num_nat, *self.target_shape))
            self.assertTrue(np.sum( msg.mask) == self.gen_batch)
            self.assertTrue(np.sum(~msg.mask) == num_nat)

    def test_errors(self):
        
        # Wrong target shape
        generator = self._get_generator_nat(
            num_images=100,
            batch_size=12,
            target_shape=(3, 242, 242) # wrong!
        )
        
        with self.assertRaises(ValueError):
            self.run_mock_inp_with_nat(generator, mask=[True]*self.gen_batch + [False]*10)
            
        # Mask but no dataloader
        generator = InverseAlexGenerator(
            root=self.root,
            variant='fc8'
        ).to(device)
        
        with self.assertRaises(AssertionError):
            self.run_mock_inp_with_nat(generator, mask=[True]*self.gen_batch + [False]*10)
