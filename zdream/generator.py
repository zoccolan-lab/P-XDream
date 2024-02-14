from pathlib import Path
import os
import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange
from abc import abstractmethod
from PIL import Image
from diffusers.models.unets.unet_2d import UNet2DModel

from functools import partial
from typing import List, Dict, cast, Callable

from .utils import lazydefault
from .utils import multichar_split
from .utils import multioption_prompt


class Generator(nn.Module):
    '''
    Base class for generic generators. A generator
    implements a `generate` method that converts latent
    codes (i.e. parameters from optimizers) into images.

    Generator are also responsible for tracking the
    history of generated images.
    '''

    def __init__(
        self,
        name : str,
    ) -> None:
        
        self.name = name

        # List for tracking image history
        self._im_hist : List[Image.Image] = []

    @abstractmethod
    def load(self, path : str | Path) -> None:
        '''
        '''
        pass

    @abstractmethod
    def forward(self):
        '''
        '''
        pass

# TODO: This is @Lorenzo's job!
class InverseAlexGenerator(Generator):

    def __init__(
        self,
        root : str,
        variant : str | None = 'fc8',
    ) -> None:
        super().__init__(name='inv_alexnet')

        # Get the networks paths based on provided root folder
        nets_path = self._get_net_paths(base_nets_dir=root)

        # If variant is not provided at initialization, we ask the experimenter
        # for generator variant of choice (from option list).
        user_in = cast(
                Callable[[], str],
                partial(
                    multioption_prompt,
                    opt_list=list(nets_path.keys()),
                    in_prompt='select your generator:',
                )
            )
        variant = lazydefault(variant, user_in)
        
        # Build the network layers based on provided generator variant
        self.layers = self._build(variant)

        # Load the corresponding checkpoint
        self.load(nets_path[variant])

        # Put the generator in evaluate mode by default
        self.eval()

    def load(self, path : str | Path) -> None:
        '''
        Load generator neural networks weights from file.

        Args:
        - path (str): Path to the network weights.
        '''
        
        self.layers.load_state_dict(
            torch.load(path, map_location='cuda')
        )
        
    @torch.no_grad()
    def forward(self, x):
        x = self.layers(x)

        # TODO: @Lorenzo Why this scaling here? Kreimann does it in his code. I don't know why this is
        if self.type_net in ['conv','norm']:
            x = x * 255

        return x  

    def _build(self, variant : str = 'fc8') -> nn.Module:
        # Get type of network (i.e: norm, conv, pool, fc)
        self.type_net = multichar_split(variant)[0][:-1]

        if   variant == 'fc8': num_inputs = 1000
        elif variant == 'fc7': num_inputs = 4096
        elif variant == 'fc6': num_inputs = 4096
        if   variant == 'norm1': l1_ios = ( 96, 128, 3, 2)
        elif variant == 'norm2': l1_ios = (256, 256, 3, 1)

        templates = {
            'fc'   : nn.Sequential(
                    nn.Linear(num_inputs, 4096),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Linear(4096, 4096),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Linear(4096, 4096),
                    nn.LeakyReLU(negative_slope=0.3),
                    Rearrange('b (c h w) -> b c h w', c=256, h=4, w=4),
                    nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False), 
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)
                ),
            'pool' : nn.Sequential(
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Conv2d(512, 512, 3, padding=0),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False), 
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)
                ),
            'conv' : nn.Sequential(
                    nn.Conv2d(384, 384, 3, padding=0),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Conv2d(384, 512, 3, padding=0),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Conv2d(512, 512, 2, padding=0),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False),
                    nn.Tanh()
                ),
            'norm' : nn.Sequential(
                    nn.Conv2d(*l1_ios, padding=2),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Conv2d(l1_ios[1], 128, 3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Conv2d(128, 128, 3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
                    nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False),
                    nn.Tanh()
                )
            }
        
        return templates[variant]
            
    def _get_net_paths(self, base_nets_dir : str) -> Dict[str, Path]:
        """
        Retrieves the paths of the files of the weights of pytorch neural nets within a base directory and returns a dictionary
        where the keys are the file names and the values are the full paths to those files.

        Args:
            base_nets_dir (str): The path of the base directory (i.e. the dir that contains all nn files). Default is '/content/drive/MyDrive/XDREAM'.

        Returns:
            Dict[str, str]: A dictionary where the keys are the nn file names and the values are the full paths to those files.
        """
        self.base_nets_dir = Path(base_nets_dir)
        nets_dict = {}
        for root, _, files in os.walk(base_nets_dir): #walk on the base net dir
            for f in files: #if you find files...
                if f.lower().endswith(('.pt', '.pth')): #and they are .pt/.pth
                    file_path = os.path.join(root, f) 
                    nets_dict[f] = file_path #add the files to nets_dict
        return nets_dict  

# TODO: This is @Paolo's job!
class SDXLTurboGenerator(Generator):
    '''
    '''