import torch.nn as nn
from torch import Tensor
from abc import abstractmethod
from PIL import Image
from diffusers.models.unets.unet_2d import UNet2DModel

from typing import List, Tuple


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
    '''
    '''
    
    def __init__(
        self,
        begin_layer : str
    ) -> None:
        super().__init__('inv_alexnet')
        
        # Construction of generator
        # TODO: Lorenzo do your magic here
        self.layers = nn.Sequential(
            ...
        )
        
    def forward(self, inp : Tensor) -> Tensor:
        return self.layers(inp)
    
    def load(base_nets_dir : str ='/content/drive/MyDrive/XDREAM') -> Tuple[nn.Module, nn.Module]:
        """
        Load subject and generator neural networks.

        Args:
        - base_nets_dir (str): The base directory where the neural network models are stored. Default is '/content/drive/MyDrive/XDREAM'.

        Returns:
        - Tuple[torch.nn.Module, torch.nn.Module]: A tuple containing the loaded subject neural network and generator neural network.
        """
        nets_dict = get_net_paths(base_nets_dir=base_nets_dir); subject_nets_names = ['alexnet']
        #ask the sbj for subject nn and generator of choice. for both, get nn name (for net_obj_dict) and path to weight file
        gen_keys = [key for key in nets_dict.keys() if any(sbj_nn not in key for sbj_nn in subject_nets_names) and 'base_nets_dir' not in key]
        subj_nn_keys = [key for key in nets_dict.keys() if any(sbj_nn in key for sbj_nn in subject_nets_names) and 'base_nets_dir' not in key]
        sbj_nn = multioption_prompt(opt_list=subj_nn_keys, in_prompt='select your subject neural net:'); sbj_nn_path = nets_dict[sbj_nn]
        gen = multioption_prompt(opt_list=gen_keys, in_prompt='select your generator:'); gen_path = nets_dict[gen]
        sbj_nn = multichar_split(sbj_nn)[0]; gen = multichar_split(gen)[0]


        net_obj_dict = { #si può trovare modo piu elegante di metterlo
            'norm1': DeePSiMNorm,
            'norm2': DeePSiMNorm2,
            'conv3': DeePSiMConv34,
            'conv4': DeePSiMConv34,
            'pool5': DeePSiMPool5,
            'fc6': DeePSiMFc,
            'fc7': DeePSiMFc,
            'fc8': DeePSiMFc8,
        }
        #instantiate the sbj_nn and gen objects
        sbj_nn_obj = net_obj_dict[sbj_nn]()
        gen_nn_obj = net_obj_dict[gen]()
        #load state dict for both, put both in cuda and put both in evaluation mode
        sbj_nn_obj.load_state_dict(torch.load(sbj_nn_path, map_location='cuda')); sbj_nn_obj.cuda(); sbj_nn_obj.eval()
        gen_nn_obj.load_state_dict(torch.load(gen_path, map_location='cuda')); gen_nn_obj.cuda(); gen_nn_obj.eval()
        return sbj_nn_obj, gen_nn_obj


# TODO: This is @Paolo's job!
class SDXLTurboGenerator(Generator):
    '''
    '''