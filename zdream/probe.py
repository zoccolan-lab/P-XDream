import numpy as np
import torch.nn as nn
from uuid import uuid4
from torch import Tensor
from abc import ABC, abstractmethod

from einops import rearrange
from collections import defaultdict
from typing import Dict, Tuple, List, Any
from numpy.typing import DTypeLike, NDArray

from .model import SubjectState

class SilicoProbe(ABC):
    '''
    Abstract probe to be used with an artificial neural network
    that implements a torch.hook (mostly forward_hook) via its
    __call__ function to achieve a given task (either by returning
    something or most likely via side-effects).
    '''

    def __init__(self) -> None:
        self.unique_id = uuid4()

    def __hash__(self) -> int:
        return hash(self.unique_id)
    
    def __eq__(self, other : 'SilicoProbe') -> bool:
        return self.unique_id == other.unique_id

    @abstractmethod
    def __call__(
        self,
        module : nn.Module,
        inp : Tuple[Tensor, ...],
        out : Tensor,    
    ) -> Any | None:
        '''
        Abstract implementation of PyTorch forward hook, each
        probe should provide its specific implementation to
        accomplish its given task.
        
        :param module: The calling torch Module who raised the
            hook callback
        :type module: torch.nn.Module
        :param inp: The input to the calling module
        :type inp: Tuple of torch Tensors
        :param out: The computed calling module output (callback
            is raised after at the end of the forward_step)
        :type out: Torch Tensor
        
        :returns: Possibly anything (probably discarded by torch)
        :rtype: Any or None 
        '''
        pass

    @abstractmethod
    def clean(self) -> None:
        pass

class NamingProbe(SilicoProbe):
    '''
    Simple probe whose task is to attach to a given torch Module
    a unique identifier to each of its sub-modules (layers).
    '''

    def __init__(
        self,
        attr_name : str = 'name'    
    ) -> None:
        '''
        Construct a NamingProbe by specifying the name of the
        attribute that the probe attaches to each sub-module of
        the target torch.nn.Module as its unique identifier.
        
        :param attr_name: Name of the new attribute
        :type attr_name: string
        '''
        
        super().__init__()
        
        self.attr_name = attr_name
        
        self.depth = -1
        self.occur = defaultdict(lambda : 0)

    def __call__(
        self,
        module : nn.Module,
        inp : Tuple[Tensor, ...],
        out : Tensor,
    ) -> None:
        '''
        Custom hook designed used to attach names to each layer
        in an artificial network. This function SHOULD NOT be
        called directly by the user, it should be called via
        the `forward_hook` attached to the network layer.
        '''
        
        name = module._get_name().lower()
        
        self.depth       += 1
        self.occur[name] += 1
        
        # Build a unique identifier for this module based
        # on module name, depth and occurrence of this
        # particular layer/module type
        depth = str(self.depth      ).zfill(2)
        occur = str(self.occur[name]).zfill(2)
        identifier = f'{depth}_{name}_{occur}'
        
        # Attach the unique identifier to the (sub-)module
        setattr(module, self.attr_name, identifier)
        
    def clean(self) -> None:
        '''
        Reset the depth and occurrence dictionary
        '''
        self.depth = -1
        self.occur = defaultdict(lambda : 0)

class RecordingProbe(SilicoProbe):
    '''
    Basic probe to attach to an artificial network to
    record the activations of an arbitrary number of
    hidden units in arbitrary layers. 
    '''
    
    def __init__(
        self,
        target : Dict[str, None] | Dict[str, Tuple[int, ...]] | Dict[str, Tuple[NDArray, ...]],
        format : DTypeLike = np.float32,
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
        super().__init__()
        
        self._target = target
        self._format = format
        
        # Here we define the activations dictionary of the probe.
        # The dictionary is indexed by the layer name and contains
        # a list with all the activations to which it was exposed to.
        self._data : Dict[str, List[NDArray]] = defaultdict(list)
        
    @property
    def features(self) -> SubjectState:
        '''
        Returns a dictionary of probe activations indexed by
        layer name. The activation is a tensor with first dimension
        referring to the specific activation.

        :return: _description_
        :rtype: Dict[str, NDArray]
        '''
        return {
            k : np.concatenate(v, axis=0) for k, v in self._data.items()
        }
        
    @property
    def target_names(self) -> List[str]:
        '''
        Returns probe target layer names.

        :return: Probe target names.
        :rtype: List[str]
        '''
        return list(self._target.keys())
        
    def __call__(
        self,
        module : nn.Module,
        inp : Tuple[Tensor, ...],
        out : Tensor
    ) -> None:
        '''
        Custom hook designed to record from an artificial network. This
        function SHOULD NOT be called directly by the user, it should be
        called via the `forward_hook` attached to the network layer.
        This function stores the layer outputs in the data dictionary.
        Function signature is the one expected by the hook mechanism.
        
        NOTE: This function assumes! the module posses the attribute
            `name` which is a unique string identifier for this layer.
            Please use the dedicated NamingProbe to properly attach
            names to each layer in the targeted artificial network.
        
        :param module: Current network layer we are registering from.
        :type module: torch.nn.Module
        :param inp: The torch Tensor the layer received as input
        :type inp: Tuple of torch.Tensor
        :param out: The torch Tensor the layer produced as output
        :type out: torch.Tensor
        
        :returns: None, data is stored as a side-effect in the class data
            attribute that can be inspected at subsequent times.
        :rtype: None
        '''
        if not hasattr(module, 'name'):
            raise AttributeError(f'Encounter module {module} with unregistered name.')
        
        # If this module is not within out target, we just skip
        if module.name not in self._target: return

        # Grab the layer output activations and put special care to
        # detach them from torch computational graph, bring them to
        # GPU and convert them to numpy for easier storage and portability
        full_act : np.ndarray = out.detach().cpu().numpy()#.squeeze()
        
        # From the whole set of activation, we extract the targeted units
        # NOTE: If None is provided as target, we simply retain the whole
        #       set of activations from this layer
        targ_idx = self._target[module.name]
        targ_act = full_act if targ_idx is None else full_act[(slice(None), *targ_idx)]
        
        # Rearrange data to have common shape [batch_size, num_units] and
        # be formatted using the desired numerical format (saving memory)
        targ_act = rearrange(targ_act.astype(self._format), 'b ... -> b (...)')
        
        # Register the network activations in probe data storage
        self._data[module.name].append(targ_act)
        
    def clean(self) -> None:
        '''
        Remove all stored activations from data storage 
        '''
        self._data = defaultdict(list)
        

# TODO Superclass Recording with (SilicoRecording, AnimalRecording)
#      with an abstract method __call__(self, Stimulus) -> SubjectState 
# class SilicoRecorder:
#     '''
#         Class representing a recording in silico from a network
#         over a set of input stimuli.
#     '''
    
#     def __init__(self, network : NetworkSubject, probe : RecordingProbe) -> None:
#         '''
#         The constructor checks consistency between network and
#         probe layers names and attach the probe hooks to the network
        
#         :param network: Network representing a tasked subject.
#         :type network: NetworkSubject
#         :param probe: Probe for recording activation.
#         :type probe: SilicoProbe
#         :param stimuli: Set of visual stimuli.
#         :type stimuli: Tensor
#         '''
        
#         # Check if probe layers exist in the network
#         assert all(
#             e in network.layer_names for e in probe.target_names 
#         ),f"Probe recording sites not in the network: {set(probe.target_names).difference(network.layer_names)}"
        
#         self._network: NetworkSubject = network
#         self._probe: RecordingProbe = probe
        
#         # Attach hooks
#         for target in probe.target_names:
#             self._network.get_layer(layer_name=target).register_forward_hook(self._probe)  # TODO callback as __call__ method of an object ?
            
#     def __call__(self, stimuli: Tensor) -> SubjectState:
#         """
#         """
        
#         self._network(stimuli)
        
#         out = self._probe.features
        
#         self._probe.clean() # TODO Make sense?
        
#         return out