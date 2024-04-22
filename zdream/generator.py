'''
This file contains the implementation of the generators used to produce synthetic images.
It provides the implementation of two generators: DeePSiM and BigGAN.
'''

import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Dict, cast, Callable, Tuple, Literal
from tqdm.auto import trange

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from einops import rearrange
from einops.layers.torch import Rearrange
from pytorch_pretrained_biggan import BigGAN

from zdream.utils.logger import Logger, SilentLogger

from .utils.types import Stimuli
from .utils.types import Codes
from .utils.misc import default, device


'''
Possible variants for the DeePSiM
They refer to the specific .pt file with trained model weights.
The weights are available at [https://drive.google.com/drive/folders/1sV54kv5VXvtx4om1c9kBPbdlNuurkGFi]
'''
DeePSIMVariant = Literal[
    'conv3', 'conv4', 'norm1', 'norm2', 'pool5', 'fc6', 'fc7', 'fc8'
]

class Generator(ABC, nn.Module):
    '''
    Base class for generic generators.
    
    A generator implements its generative logic in the `_forward()` method that converts 
    latent codes (i.e. latent representations of images) into actual visual stimuli.
    '''

    def __init__(
        self,
        name : str,
        output_pipe : Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        '''
        Create a new instance of a generator

        :param name: Generator name identifying a pretrained architecture.
        :type name: str
        :param output_pipe: Pipeline of postprocessing operation to be applied to raw generated images.
            If not specified, the raw images are returned without any transformation.
        :type output_pipe: Callable[[Tensor], Tensor] | None, optional
        '''
        
        super().__init__()
        
        self._name = name

        # If no output pipe is provided, no raw transformation is used
        self._output_pipe = default(output_pipe, cast(Callable[[Tensor], Tensor], lambda x : x))
        
        # Underlying torch module that generates images
        # NOTE: The network defaults to None. Subclasses are asked to 
        #       provide the specific instance 
        self._network = None
        
    # --- CODE MANIPULATION ---

    def find_code(
        self,
        target   : Tensor,
        num_iter : int = 500,
        rel_tol  : float = 1e-1,
        logger   : Logger = SilentLogger()
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        '''
        Optimization process to retrieve the latent code that generates a target image.
        
        This function is intended to evaluate the quality of the Generator, and to
        study it's ability 

        :param target: Target generated image.
        :type target: Tensor
        :param num_iter: Maximum number of optimization steps, defaults to 500
        :type num_iter: int, optional
        :param rel_tol: Pixel level distance tolerance to find solution, defaults to 1e-1
        :type rel_tol: float, optional
        :param logger: Logger object to print messages, defaults to SilentLogger()
        :type logger: Logger, optional
        :return: Tuple containing the latent code and the generated image and the error.
        :rtype: Tuple[Tensor, Tuple[Tensor, Tensor]]
        '''
        
        # Check if network is specified
        if not self._network:
            raise ValueError(f'Unspecified underlying network model')
        
        self._network = cast(nn.Module, self._network)

        # Extract batch size and ensure stimuli 
        # to have three dimension with unpacking
        b, _, _, _ = target.shape

        # Initialize the loss and the latent code
        loss = nn.MSELoss()
        code = torch.zeros(b, *self.input_dim, device=self.device, requires_grad=True)

        # Define the optimizer
        optim = AdamW(
            [code],
            lr=1e-3,
        )

        # Run optimization loop
        epoch = 0
        found_solution = False
        progress = trange(num_iter, desc='Code retrieval | avg. err: --- | rel. err: --- |')
        
        while not found_solution and (epoch := epoch + 1) < num_iter:
            optim.zero_grad()

            # Generate a novel set of images from the current
            # set of latent codes
            imgs = self._network(code)
            imgs = self._output_pipe(imgs)

            # Compute the loss and backpropagate
            errs : Tensor = loss(imgs, target)
            errs.backward()
            optim.step()

            # Compute the average and relative error
            a_errs = errs.mean()
            r_errs = a_errs / imgs.mean()
            
            # Update the progress bar
            p_desc = f'Code retrieval | avg. err: {a_errs:.2f} | rel. err: {r_errs:.4f}'
            progress.set_description(p_desc)
            progress.update()

            # Check if images are within the relative tolerance
            if torch.all(errs / imgs.mean() < rel_tol):
                found_solution = True
        
        progress.close()

        #If the optimization process has not found a solution within the specified
        if not found_solution:
            logger.warn('Cannot find codes within specified relative tolerance')

        return code, (imgs.detach(), errs)

    @abstractmethod
    def load(self, path : str | Path) -> None:
        '''
        Function to load the weights of the generator from a file.
        
        :param path: Path to the file containing the weights.
        :type path: str | Path
        '''
        pass

    @abstractmethod
    @torch.no_grad()
    def forward(
        self, 
        codes: Codes
    ) -> Stimuli:
        '''
        Generate stimuli from latent codes and return the stimuli.

        :param codes: Latent images code for synthetic images generation.
        :type codes: Codes
        :return: Produced stimuli set.
        :rtype: Stimuli
        '''
        
        pass
    
    # --- PROPERTIES ---
    
    @property
    @abstractmethod
    def input_dim(self) -> Tuple[int, ...]:
        '''
        Abstract property that returns the dimension of the latent code.

        :return: Dimension of the latent code.
        :rtype: Tuple[int, ...]
        '''
        pass
    
    @property
    def device(self): return next(self.parameters()).device
    ''' Return device where the generator is running. '''
    
    @property
    def dtype(self) -> torch.dtype: return next(self.parameters()).dtype
    ''' Return the data type of the generator. '''


class DeePSiMGenerator(Generator):

    def __init__(
        self,
        root : str,
        variant : DeePSIMVariant = 'fc8',
        output_pipe : Callable[[Tensor], Tensor] | None = None
    ) -> None:
        '''
        Create a new instance of a DeePSiM generator.
        
        See: Generating Images with Perceptual Similarity Metrics based on Deep Networks [https://arxiv.org/abs/1602.02644]
        
        :param root: Path to the root folder containing the pretrained network weights.
        :type root: str
        :param variant: DeepSiM variant to use, i.e. the version of the latent code. Defaults to 'fc8'.
        :type variant: DeePSIMVariant, optional
        :param output_pipe: Pipeline of postprocessing operation to be applied to raw generated images.
        :type output_pipe: Callable[[Tensor], Tensor] | None, optional
        '''
        
        # Get the networks paths based on provided root folder
        nets_path = self._get_net_paths(base_nets_dir=root)
        self._variant = variant
        
        # If not provided the default output pipe depends on the variant
        output_pipe = default(output_pipe, self._get_pipe(self._variant))
        
        super().__init__(
            name='DeePSiM',
            output_pipe=output_pipe
        )
        
        # Build the network layers based on provided generator variant
        # and load it's checkpoint
        self._network = self._build(self._variant)
        self.load(nets_path[self._variant])

        # Put the generator in evaluate mode
        self.eval()
        
    # --- STRING REPRESENTATION ---
        
    def __str__(self) -> str:
        ''' Return a string representation of the generator. '''
        return f'DeePSiMGenerator[variant: {self._variant}; in-dim: {self.input_dim}; out-dim: {self.output_dim}]'
    
    def __repr__(self) -> str: return str(self)
    ''' Returns a string representation of the generator. '''
    
    # --- LOADING ---

    def load(self, path : str | Path) -> None:
        '''
        Load generator neural networks weights from file.

        :param path: Path to the file containing the weights.
        :type path: str | Path
        '''
        
        self._network.load_state_dict(torch.load(path, map_location=self.device))
    
    def _get_net_paths(self, base_nets_dir : str) -> Dict[str, Path]:
        '''
        Retrieves the paths of the files of the weights of TORCH neural nets within a base directory. 
        
        It returns a dictionary where the keys are the file names  and the values are the full paths to those files.

        :param base_nets_dir: Path to the root folder containing the pretrained network weights.
        :type base_nets_dir: str
        :return: Dictionary containing the paths to the weights of the networks.
        :rtype: Dict[str, Path]
        '''
        
        # Get the paths of the files with the weights of the networks
        root = Path(base_nets_dir)
        
        # Get the paths of the files with the weights of the networks
        nets_dict = {
            Path(file).stem : Path(base, file)
            for base, _, files in os.walk(root)
            for file in files if file.endswith(('.pt', 'pth'))
        }
        
        return nets_dict 
    
    # --- IMAGE GENERATION ---
        
    @torch.no_grad()
    def forward(
        self, 
        codes : Codes,
    ) -> Stimuli:
        '''
        Generated synthetic images starting with their latent code

        :param codes: Latent images code for synthetic images generation.
        :type codes: Tensor | NDArray
        :return: Generated stimuli set.
        :rtype: Stimuli
        '''
        
        # NOTE: We convert numpy codes to tensors as input for the generator
        codes_ = torch.from_numpy(codes).to(self.device).to(self.dtype)
            
        # Generate the synthetic images and apply the output pipe
        gens = self._network(codes_)
        gens = self._output_pipe(gens)

        # Dimension conversion
        if self.type_net in ['conv', 'norm']:
            gens *= 255

        return gens
    
    # --- PROPERTIES ---

    @property
    def input_dim(self) -> Tuple[int, ...]:
        ''' Return the dimension of the latent code depending on the Generator variant. '''
        
        match self._variant:
            case 'fc8':   return (1000,)
            case 'fc7':   return (4096,)
            case 'fc6':   return (4096,)
            case 'conv3': return (384, 13, 13)
            case 'conv4': return (384, 13, 13)
            case 'norm1': return (96, 27, 27)
            case 'norm2': return (256, 13, 13)
            case 'pool5': return (256, 6, 6)
            case _: raise ValueError(f'Unsupported variant {self._variant}')

    @property
    def output_dim(self) -> Tuple[int, int, int]:
        ''' Return the dimension of the output image depending on the Generator variant. '''
        
        match self._variant:
            case 'norm1': return (3, 240, 240)
            case 'norm2': return (3, 240, 240)
            case _      : return (3, 256, 256)
            
    def _get_pipe(self, variant : DeePSIMVariant) -> Callable[[Tensor], Tensor]:
        ''' Return the output pipe for the generator variant. '''
        
        def _opt1(imgs : Tensor) -> Tensor:
            ''' Default output pipe for the generator.'''
            
            mean = torch.tensor((104.0, 117.0, 123.0), device=imgs.device)
            mean = rearrange(mean, 'c -> c 1 1')
            imgs = imgs + mean
            imgs = imgs / 255.

            return imgs.clamp(0, 1)
        
        def _opt2(imgs : Tensor) -> Tensor:
            ''' Output pipe for norm and conv variants of the generator.'''
            
            return 0.5 * (1 + imgs)
        
        match variant:    
            case 'norm1' | 'norm2': return _opt2
            case 'conv3' | 'conv4': return _opt2
            case _: return _opt1 
            
    # --- NETWORK ARCHITECTURE ---

    def _build(self, variant : str) -> nn.Module:
        '''
        Build the network architecture based on the variant of the generator.

        :param variant: Variant of the generator.
        :type variant: str
        '''

        # Get type of network (i.e: norm, conv, pool, fc)
        # by separating the layer name from unit count
        self.type_net, _ = re.match(r'([a-zA-Z]+)(\d+)', variant).groups() # type: ignore

        match variant:
            case 'fc8'  : num_inputs = 1000
            case 'fc7'  : num_inputs = 4096
            case 'fc6'  : num_inputs = 4096
            case 'norm1': inp_par    = ( 96, 128, 3, 2)
            case 'norm2': inp_par    = (256, 256, 3, 1)
            case _: pass
            
        templates = {
            'fc'   : lambda : nn.Sequential(OrderedDict([
                    ('fc7',       nn.Linear(num_inputs, 4096)),
                    ('lrelu01',   nn.LeakyReLU(negative_slope=0.3)),
                    ('fc6',       nn.Linear(4096, 4096)),
                    ('lrelu02',   nn.LeakyReLU(negative_slope=0.3)),
                    ('fc5',       nn.Linear(4096, 4096)),
                    ('lrelu03',   nn.LeakyReLU(negative_slope=0.3)),
                    ('rearrange', Rearrange('b (c h w) -> b c h w', c=256, h=4, w=4)),
                    ('tconv5_0',  nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)),
                    ('lrelu04',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv5_1',  nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False)),
                    ('lrelu05',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_0',  nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)),
                    ('lrelu06',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_1',  nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)), 
                    ('lrelu07',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_0',  nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)),
                    ('lrelu08',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_1',  nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)),
                    ('lrelu09',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv2',    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)),
                    ('lrelu10',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv1',    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)),
                    ('lrelu11',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv0',    nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)),
                ])
            ),
            'pool' : lambda : nn.Sequential(OrderedDict([
                    ('conv6',    nn.Conv2d(256, 512, 3, padding=1)),
                    ('lrelu01',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv7',    nn.Conv2d(512, 512, 3, padding=1)),
                    ('lrelu02',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv8',    nn.Conv2d(512, 512, 3, padding=0)),
                    ('lrelu03',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv5_0', nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)),
                    ('lrelu04',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv5_1', nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False)),
                    ('lrelu05',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_0', nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)),
                    ('lrelu06',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_1', nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)), 
                    ('lrelu07',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_0', nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)),
                    ('lrelu08',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_1', nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)),
                    ('lrelu09',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv2',   nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)),
                    ('lrelu10',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv1',   nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)),
                    ('lrelu11',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv0',   nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)),
                ])
            ),
            'conv' : lambda : nn.Sequential(OrderedDict([
                    ('conv6',    nn.Conv2d(384, 384, 3, padding=0)),
                    ('lrelu01',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv7',    nn.Conv2d(384, 512, 3, padding=0)),
                    ('lrelu02',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv8',    nn.Conv2d(512, 512, 2, padding=0)),
                    ('lrelu03',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv5_0', nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)),
                    ('lrelu04',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv5_1', nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)),
                    ('lrelu05',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_0', nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)),
                    ('lrelu06',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_1', nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)),
                    ('lrelu07',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_0', nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False)),
                    ('lrelu08',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_1', nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)),
                    ('lrelu09',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv2_0', nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)),
                    ('lrelu10',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv2_1',  nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False)),
                    ('lrelu11',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv1_0', nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False)),
                    ('lrelu12',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv1_1',  nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False)),
                    ('tanh',     nn.Tanh()),
                ])
            ),
            'norm' : lambda : nn.Sequential(OrderedDict([
                    ('conv6',    nn.Conv2d(*inp_par, padding=2)),
                    ('lrelu1',   nn.LeakyReLU(negative_slope=0.3)),
                    ('conv7',    nn.Conv2d(inp_par[1], 128, 3, stride=1, padding=1)),
                    ('lrelu2',   nn.LeakyReLU(negative_slope=0.3)),
                    ('conv8',    nn.Conv2d(128, 128, 3, stride=1, padding=1)),
                    ('lrelu3',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_0', nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False)),
                    ('lrelu4',   nn.LeakyReLU(negative_slope=0.3)),
                    ('conv4_1',  nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)),
                    ('lrelu5',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_0', nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)),
                    ('lrelu6',   nn.LeakyReLU(negative_slope=0.3)),
                    ('conv3_1',  nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                    ('lrelu7',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv2_0', nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)),
                    ('lrelu8',   nn.LeakyReLU(negative_slope=0.3)),
                    ('conv2_1',  nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)),
                    ('lrelu9',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv1_0', nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False)),
                    ('conv1_1',  nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False)),
                    ('tanh',     nn.Tanh()),
                ]))
            }
        
        return templates[self.type_net]() 


class BigGANGenerator(Generator):
    '''
    Generator using pretrained BigGAN from PYTORCH.
    '''
    
    CLASS_VECTOR = 1000
    ''' Number of classes in the BigGAN model.'''
    
    NOISE_VECTOR =  128
    ''' Length of the noise vector in the BigGAN model.'''
    
    NAME = 'biggan-deep-256'
    ''' Model name '''
    
    
    def __init__(
        self, 
        output_pipe: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        '''
        Create a new instance of a BigGAN generator.

        :param output_pipe: Pipeline of postprocessing operation to be applied to raw generated images.
        :type output_pipe: Callable[[Tensor], Tensor] | None, optional
        '''
        
        super().__init__(
            name='biggan-deep-256', 
            output_pipe=output_pipe, 
        )
        
        self.load(path=self.NAME)
        self._model.to(device)
        
    # --- LOADING ---
        
    def load(self, path : str | Path) -> None:
        '''
        Load the BigGAN model from a file.

        :param path: Path to the file containing the weights.
        :type path: str | Path
        '''
    
        self._model = BigGAN.from_pretrained(path)
        
    # --- DIMENSIONS PROPERTIES ---
    
    @property
    def input_dim(self) -> Tuple[int, ...]:
        '''
        We use the concatenated class and noise vectors as a unique code.

        :return: Dimension of the latent code.
        :rtype: Tuple[int, ...]
        '''
        
        return ((self.CLASS_VECTOR + self.NOISE_VECTOR), )
    
    
    @property
    def output_dim(self) -> Tuple[int, ...]: return (3, 256, 256)
    
    # --- IMAGE GENERATION ---
    
    def forward(
        self, 
        codes: Codes
    ) -> Stimuli:
        '''
        Generate stimuli from latent codes and return the stimuli.

        :param codes: Latent images code for synthetic images generation.
        :type codes: Codes
        :return: Produced stimuli set.
        :rtype: Stimuli
        '''
        
        # TODO Control vector with input parameters
        t            = 0.4
        noise_vector =      torch.tensor(codes[:, :self.NOISE_VECTOR], dtype=torch.float32, device=device)
        class_vector = .5 * torch.tensor(codes[:, self.NOISE_VECTOR:], dtype=torch.float32, device=device)

        # Generate an image
        with torch.no_grad():
            stimuli = self._model(noise_vector, class_vector, t)
            
        # TODO Put in output pipeline
        stimuli += 1
        stimuli *= 0.5
            
        return stimuli
