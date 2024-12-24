# Set of functions used for loading parameters for different models
import torch
from torch import nn
from torchvision.models import get_model, get_model_weights

from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model

from robustbench import load_model

from pxdream.utils.misc import InputLayer
from experiment.utils.args import DATASET

def torch_load(net_sbj: 'TorchNetworkSubject', weights_path: str = '', pretrained: bool = True):
    """
    Load weights into a neural network subject (net_sbj) and initialize its network architecture.

    :param net_sbj: The neural network subject.
    :type net_sbj: TorchNetworkSubject (NOTE: PROBLEM OF CIRCULAR IMPORT)
    :param weights_path: The file path to the pre-trained weights. If not provided, the function will either load weights from the torchvision hub or initialize with random weights based on the 'pretrained' flag.
    :type weights_path: str, optional
    :param pretrained: If True, loads the model with pre-trained weights from the torchvision hub. If False, initializes the model with random weights.
    :type pretrained: bool, optional
    :returns: None
    """
    
    if weights_path != '':
        net_sbj._weights = torch.load(weights_path)
    else:
        # Load the torch model via its name from the torchvision hub if pretrained
        # otherwise initialize it with random weights (weights=None).
        net_sbj._weights = get_model_weights(net_sbj._name).DEFAULT if pretrained else None  # type: ignore

    net_sbj._network = nn.Sequential(
        InputLayer(),
        get_model(net_sbj._name, weights=net_sbj._weights)
    ).to(net_sbj._device)
    # Apply preprocessing associated to pretrained weights
    # NOTE: `self.weights` is None in the case of random initialization
    #       and corresponds to no transformation
    net_sbj._preprocessing = net_sbj._weights.transforms() if pretrained else lambda x: x
    
    #return net_sbj
    
def madryLab_robust_load(net_sbj: 'TorchNetworkSubject', weights_path: str, pretrained: bool = False):
    """
    Load weights into a neural network subject (net_sbj) using MadryLab's robustness library 
    and initialize its network architecture.

    :param net_sbj: The neural network subject.
    :type net_sbj: TorchNetworkSubject (NOTE: PROBLEM OF CIRCULAR IMPORT)
    :param weights_path: The file path to the pre-trained weights.
    :type weights_path: str
    :param pretrained: unused flag, kept for compatibility with other loading
        functions in torch_net_load_functs.py.
    :type pretrained: bool, optional
    :returns: None
    """
    # loading robust models used during BMM summer school
    ds = ImageNet(DATASET)
    model, _ = make_and_restore_model(arch=net_sbj._name, dataset=ds, resume_path=weights_path)
    net_sbj._network = nn.Sequential(
        InputLayer(),
        model.model
    ).to(net_sbj._device)
    
    net_sbj._preprocessing = model.normalizer
    # this loading implies working with robust models
    net_sbj.robust = True

def robustBench_load(net_sbj: 'TorchNetworkSubject', weights_path: str, pretrained: bool = False):
    """
    Load weights into a neural network subject (net_sbj) using Robust bench robustness library 
    and initialize its network architecture.

    :param net_sbj: The neural network subject.
    :type net_sbj: TorchNetworkSubject (NOTE: PROBLEM OF CIRCULAR IMPORT)
    :param weights_path: The name of the pre-trained network.
    :type weights_path: str
    :param pretrained: unused flag, kept for compatibility with other loading
        functions in torch_net_load_functs.py.
    :type pretrained: bool, optional
    :returns: None
    """
    model = load_model(model_name=weights_path, dataset='imagenet')
    net_sbj._network = nn.Sequential(
        InputLayer(),
        model.model
    ).to(net_sbj._device)
    
    net_sbj._preprocessing = model.normalize.to(net_sbj._device)
    # this loading implies working with robust models
    net_sbj.robust = True 
    
    
    
    
NET_LOAD_DICT ={
    'torch_load': torch_load,
    'madryLab_robust_load': madryLab_robust_load,
    'robustBench_load': robustBench_load
}