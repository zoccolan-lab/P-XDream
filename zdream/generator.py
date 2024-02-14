from pathlib import Path
import os
import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange
from abc import abstractmethod
from PIL import Image
from diffusers.models.unets.unet_2d import UNet2DModel

from typing import List, Tuple

from zdream.utils import *


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
    def load(self, path : str) -> None:
        '''
        
        '''

    @abstractmethod
    def forward(self):
        '''
        '''
        pass

# TODO: This is @Lorenzo's job!
class InverseAlexGenerator(Generator):

    def __init__(self, base_nets_dir='/content/drive/MyDrive/XDREAM'):
        super().__init__()
        self.load(self, base_nets_dir = base_nets_dir)

    def build(self):
        architectures_dict = {'fc':nn.Sequential(nn.Linear(self.num_inputs, 4096),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=0.3),
            Rearrange('(b c) h w -> b c h w'),
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
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)),
            'pool':nn.Sequential(nn.Conv2d(256, 512, 3, padding=1),
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
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)),
            'conv': nn.Sequential(
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
            nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),#
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False),
            nn.Tanh()),
            'norm': nn.Sequential(nn.Conv2d(self.l1_ios[0], self.l1_ios[1], 3, stride=self.l1_ios[2], padding=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(self.l1_ios[1], 128, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
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
            nn.Tanh())}
        
        self.layers = architectures_dict[self.type_net]
    
    def get_net_paths(self, base_nets_dir):
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

    def load(self, base_nets_dir: str ='/content/drive/MyDrive/XDREAM') -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Load subject and generator neural networks.

        Args:
        - base_nets_dir (str): The base directory where the neural network models are stored. Default is '/content/drive/MyDrive/XDREAM'.

        Returns:
        - Tuple[torch.nn.Module, torch.nn.Module]: A tuple containing the loaded subject neural network and generator neural network.
        """
        nets_dict = self.get_net_paths(base_nets_dir=base_nets_dir); 
        #ask the experimenter for subject nn and generator of choice. for both, get nn name (for net_obj_dict) and path to weight file
        gen = multioption_prompt(opt_list=nets_dict.keys(), in_prompt='select your generator:'); gen_path = nets_dict[gen]
        gen = multichar_split(gen)[0]
        self.num_inputs = 4096; self.l1_ios = (96, 128, 2)
        self.type_net = gen[:-1] #norm, conv, pool, fc
        if gen=='fc8':
            self.num_inputs = 1000
        if gen=='norm2':
            self.l1_ios = (96, 128, 2)

        self.build() #generate the appropriate architecture
        self.load_state_dict(torch.load(gen_path, map_location='cuda')), self.cuda(); self.eval()

    def forward(self, x):
        x = self.layers(x)
        if self.type_net in ['conv','norm']:
            x = x*255
        return x    

# TODO: This is @Paolo's job!
class SDXLTurboGenerator(Generator):
    '''
    '''