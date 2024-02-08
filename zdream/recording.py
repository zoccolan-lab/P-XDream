import numpy as np
import torch.nn as nn
from torch import Tensor

from collections import defaultdict
from typing import Any, Dict, Tuple

class SilicoProbe:
    '''
        Basic probe to attach to an artificial network to
        record the activations of an arbitrary number of
        hidden units in arbitrary layers. 
    '''
    
    def __init__(
        self,
        target : Dict[str, None | Tuple[int, ...] | Tuple[np.ndarray, ...]],
        format : np.dtype = np.float32,
    ) -> None:
        '''
        Artificial probe for recording hidden unit activations
        
        :param target: Specification of which units to record from which
            layer. Layers are identified via their name and units by their
            position in the layer (multi-dimensional index). If None is
            provided as specification it is assumed that ALL unit from that
            given layer are to be recorded.
        :type target: Dict[str, None | Tuple[int, ...] | Tuple[np.ndarray, ...]]
        :param format: Numeric format to use to store the data. Useful to
            reduce file size or memory footprint for large recordings
        :type format: np.dtype
        '''
        
        self.target = target
        self.format = format
        
        # Here we define the activations dictionary of the probe.
        # The dictionary is indexed by the layer name and contains
        # a list with all the activations to which it was exposed to.
        self.data = defaultdict(list)
        
    def __call__(
        self,
        module : nn.Module,
        inp : Tensor,
        out : Tensor
    ) -> None:
        '''
        Custom hook designed to record from an artificial network. This
        function SHOULD NOT be called directly by the user, it should be
        called via the `forward_hook` attached to the network layer.
        This function stores the layer outputs in the data dictionary.
        Function signature is the one expected by the hook mechanism.
        
        NOTE: This function assumes! the module posses the attribute
            `name` which is a unique string identifier for this layer
        
        :param module: Current network layer we are registering from.
        :type module: torch.nn.Module
        :param inp: The torch Tensor the layer received as input
        :type inp: torch.Tensor
        :param out: The torch Tensor the layer produced as output
        :type out: torch.Tensor
        
        :returns: None, data is stored as a side-effect in the class data
            attribute that can be inspected at subsequent times.
        :rtype: None
        '''
        if not hasattr(module, 'name'):
            raise AttributeError(f'Encounter module {module} with unregistered name.')
        
        # Grab the layer output activations and put special care to
        # detach them from torch computational graph, bring them to
        # GPU and convert them to numpy for easier storage and portability
        full_act : np.ndarray = out.detach().cpu().numpy().astype(self.format).squeeze()
        
        # From the whole set of activation, we extract the targeted units
        # NOTE: If None is provided as target, we simply retain the whole
        #       set of activations from this layer
        targ_idx = self.target[module.name]
        targ_act = full_act if targ_idx is None else full_act[(slice(None), *targ_idx)]
        
        # Register the network activations in probe data storage
        self.data[module.name].append(targ_act)
        
    def empty(self) -> None:
        '''
        Remove all stored activations from data storage 
        '''
        self.data = defaultdict(list)