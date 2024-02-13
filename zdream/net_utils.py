from pathlib import Path
import os
import torch
from typing import Tuple

from zdream.utils import *
from zdream.networks import *
from zdream.generators import *

    
def get_net_paths(base_nets_dir='/content/drive/MyDrive/XDREAM'):
    """
    Retrieves the paths of the files of the weights of pytorch neural nets within a base directory and returns a dictionary
    where the keys are the file names and the values are the full paths to those files.

    Args:
        base_nets_dir (str): The path of the base directory (i.e. the dir that contains all nn files). Default is '/content/drive/MyDrive/XDREAM'.

    Returns:
        Dict[str, str]: A dictionary where the keys are the nn file names and the values are the full paths to those files.
    """
    nets_dict = {'base_nets_dir':Path(base_nets_dir)}
    for root, _, files in os.walk(base_nets_dir): #walk on the base net dir
        for f in files: #if you find files...
          if f.lower().endswith(('.pt', '.pth')): #and they are .pt/.pth
            file_path = os.path.join(root, f) 
            nets_dict[f] = file_path #add the files to nets_dict
    return nets_dict

def load_networks(base_nets_dir: str ='/content/drive/MyDrive/XDREAM') -> Tuple[torch.nn.Module, torch.nn.Module]:
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
        'alexnet': AlexNet,
    }
    #instantiate the sbj_nn and gen objects
    sbj_nn_obj = net_obj_dict[sbj_nn]()
    gen_nn_obj = net_obj_dict[gen]()
    #load state dict for both, put both in cuda and put both in evaluation mode
    sbj_nn_obj.load_state_dict(torch.load(sbj_nn_path, map_location='cuda')); sbj_nn_obj.cuda(); sbj_nn_obj.eval()
    gen_nn_obj.load_state_dict(torch.load(gen_path, map_location='cuda')); gen_nn_obj.cuda(); gen_nn_obj.eval()
    return sbj_nn_obj, gen_nn_obj




