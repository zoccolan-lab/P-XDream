# Set of functions used for loading parameters for different models
import torch
from torch import nn
from torchvision.models import get_model, get_model_weights

from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model

from pxdream.utils.misc import InputLayer
from experiment.utils.args import DATASET

def torch_load(net_sbj  , weights_path: str = '', pretrained: bool = False):
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
    
def madryLab_robust_load(net_sbj, weights_path: str = '', pretrained: bool = False):
    # loading robust models used during BMM summer school
    ds = ImageNet(DATASET)
    model, _ = make_and_restore_model(arch=net_sbj._name, dataset=ds, resume_path=weights_path)
    net_sbj._network = nn.Sequential(
        InputLayer(),
        model.model
    ).to(net_sbj._device)
    
    net_sbj._preprocessing = model.normalizer
    
    
NET_LOAD_DICT ={
    'torch_load': torch_load,
    'madryLab_robust_load': madryLab_robust_load
}