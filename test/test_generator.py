'''
Collection of codes for testing the workings of zdream generators
'''

from os import path
import random
import unittest
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from zdream.model import Mask
from zdream.model import Stimuli
from zdream.generator import InverseAlexGenerator
from zdream.model import Message
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from zdream.utils import device, read_json

# Loading `local_settings.json` for custom local settings
test_folder = path.dirname(path.abspath(__file__))
test_settings_fp = path.join(test_folder, 'local_settings.json')
test_settings: Dict[str, Any] = read_json(path=test_settings_fp)

class _RandomImageDataset(Dataset):
    '''
    Random image dataset to simulate natural images to be interleaved
    in the stimuli with synthetic ones.
    '''
    
    def __init__(self, n_img: int, img_size: Tuple[int, ...]):
        self.n_img = n_img
        self.image_size = img_size
    
    def __len__(self):
        return self.n_img
    
    def __getitem__(self, idx) -> Tensor:
        
        # Simulate finite dataset
        if idx < 0 or idx >= len(self): raise ValueError(f"Invalid image idx: {idx} not in [0, {len(self)})")

        rand_img = torch.tensor(np.random.rand(*self.image_size), dtype=torch.float32)
        
        return rand_img

class InverseAlexGeneratorTest(unittest.TestCase):
    
    rand_seed = 123
    num_gen = 10
    
    # NOTE: One must edit the `local_settings.json` in test dir
    #       accordingly to proper local machine path
    root = test_settings['inverse_alex_net']

    def setUp(self) -> None:
        
        super().setUp()
        
        # Seed for pseudo random generation for 
        # random image generation and mask shuffling
        random.seed = self.rand_seed
        np.random.seed = self.rand_seed
        
    def run_mock_inp(
            self, 
            generator : InverseAlexGenerator, 
            mask: Mask | None = None
        ) -> Tuple[Stimuli, Message]:
        
        # Test generator on mock input with an optional mask
        inp_dim = generator.input_dim
        mock_inp = torch.randn(self.num_gen, *inp_dim, device=generator.device)
        
        return generator(mock_inp, mask)

    def _get_generator_with_nat_img(
        self,
        n_img : int, 
        batch_size : int
    ) -> InverseAlexGenerator:
        
        # Create a generator with a data loader, for which  
        # number of images, images size and batch size are required.
        
        generator = InverseAlexGenerator(
            root=self.root,
            variant='fc8'
        ).to(device)
        
        dataset = _RandomImageDataset(
            n_img=n_img,
            img_size=generator.output_dim
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size
        )
        
        generator.set_nat_img_loader(nat_img_loader=dataloader)
        
        return generator
    
    def test_loading_fc8(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='fc8'
        ).to(device)
        
        out, msg = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.num_gen, *generator.output_dim))
        self.assertTrue(all(msg.mask))
    
    def test_loading_fc7(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='fc7'
        ).to(device)
        
        out, msg = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.num_gen, *generator.output_dim))
        self.assertTrue(all(msg.mask))

    def test_loading_conv(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='conv4'
        ).to(device)

        out, msg = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.num_gen, *generator.output_dim))
        self.assertTrue(all(msg.mask))

    def test_loading_norm1(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='norm1'
        ).to(device)
        
        out, msg = self.run_mock_inp(generator)

        self.assertEqual(out.shape, (self.num_gen, *generator.output_dim))
        self.assertTrue(all(msg.mask))

    
    def test_loading_norm2(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='norm2'
        ).to(device)
        
        out, msg = self.run_mock_inp(generator)
        
        self.assertEqual(out.shape, (self.num_gen, *generator.output_dim))
        self.assertTrue(all(msg.mask))
    
    def test_loading_pool(self):
        generator = InverseAlexGenerator(
            root=self.root,
            variant='pool5'
        ).to(device)
        
        out, msg = self.run_mock_inp(generator)
    
        self.assertEqual(out.shape, (self.num_gen, *generator.output_dim))
        self.assertTrue(all(msg.mask))
        
    def test_natural_images(self):
        
        # We run multiple versions of the generator with natural images
        # by specifying different combination of parameters.
        
        # Triples (n_img, num_nat, dataloader_batch)
        parameters = [
            (100, 10,  1), # single batch
            (100, 10,  2), # batch divisible for the natural images
            (100, 10,  3), # batch non divisible for the natural images
            ( 1,   2,  1), # circular dataset
            ( 10, 10, 10), # same batch size as dataset
            ( 10, 20, 10), # more natural images than ones in the dataset
            ( 10,  0, 10), # no natural images
        ]
        
        for n_img, num_nat, dataloader_batch in parameters:
                
            # Create a generator
            generator = self._get_generator_with_nat_img(
                n_img=n_img,
                batch_size=dataloader_batch
            )
            
            # Create a correct mask and shuffle (random seed is set)
            mask = [True]*self.num_gen + [False]*num_nat
            random.shuffle(mask)
            
            # Run a mock input
            out, msg = self.run_mock_inp(
                generator=generator,
                mask=mask
            )
        
            # Check if 1) the output shape matches, 2) the mask is preserved,
            #          3) the flags in the mask match number of synthetic and natural images
            self.assertEqual(out.shape, (self.num_gen+num_nat, *generator.output_dim))
            self.assertTrue(list(msg.mask) == mask)
            self.assertTrue(np.sum( msg.mask) == self.num_gen)
            self.assertTrue(np.sum(~msg.mask) == num_nat)

    def test_no_dataloader(self):
        
        # We check an error is raised if the mask requires 
        # natural images but no dataloader is provided
        
        generator_no_dataloader = InverseAlexGenerator(
            root=self.root,
            variant='fc8'
        ).to(device)
        
        with self.assertRaises(AssertionError):
            self.run_mock_inp(generator_no_dataloader, mask=[True]*self.num_gen + [False]*10)
        
    def test_wrong_target(self):
        
        # We check an error is raised in the case natural
        # and synthetic images have different shapes.
        
        generator = InverseAlexGenerator(
            root=self.root,
            variant='fc8'
        ).to(device)
        
        dataset = _RandomImageDataset(
            n_img=100,
            img_size=(3, 224, 244) # wrong
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=3
        )
        
        generator.set_nat_img_loader(nat_img_loader=dataloader)
        
        with self.assertRaises(ValueError):
            self.run_mock_inp(generator, mask=[True]*self.num_gen + [False]*10)
    
    def test_true_mask(self):
        
        # We check the two conditions providing a True-only mask
        
        generator = self._get_generator_with_nat_img(
            n_img=100,
            batch_size=12
        )   
        
        # In the case the number of True doesn't match the number
        # of synthetic images an error is expected
        with self.assertRaises(ValueError):
            self.run_mock_inp(generator, mask=[True]*(self.num_gen-1))
            
        # In the case the number of True matches the number of synthetic images
        # we expect the exact behavior as if the masking was not provided  
        
        mock_inp = torch.randn(self.num_gen, *generator.input_dim, device=generator.device)
        
        stimuli_1, _ = generator(mock_inp, [True]*(self.num_gen))
        stimuli_2, _ = generator(mock_inp)
        
        self.assertTrue(torch.allclose(stimuli_1, stimuli_2, atol=1e-4))
        
    def test_inconsistent_mask(self):
        
        # In the case the True flags doesn't match the number of 
        # generated images, an error is raised.
        
        generator = self._get_generator_with_nat_img(
            n_img=100,
            batch_size=12
        ) 
        
        # True and False mask with incorrect number of True
        with self.assertRaises(ValueError):
            self.run_mock_inp(generator, mask=[True]*(self.num_gen-1) + [False]*10)
