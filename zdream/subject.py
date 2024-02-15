import torch
import warnings
from abc import ABC, abstractmethod
from torch import nn, Tensor
from typing import List, Tuple
from torchvision import models
from torch.utils.hooks import RemovableHandle

from .probe import SilicoProbe
from .probe import NamingProbe
from .probe import RecordingProbe

from .utils import unpack
from .utils import SubjectState

class Subject(ABC):
    '''
        Abstract class representing a subject (animal or network)
        tasked with a visual stimuli and generating a set of
        activations for multiple layers.
        
    '''
    
    @abstractmethod
    def foo(self):
        # TODO discuss what is a "Subject"
        pass
    
    
    
class NetworkSubject(Subject, nn.Module):
    '''
        Class representing an artificial network involved in
        a visual task experiment as an artificial counterpart 
        of an animal. A network subject extends a torch Module
        by providing a unique identifier to each layer (via a
        NamingProbe) that can later be used for recording         
    '''
    
    def __init__(
        self,
        network_name : str,
        record_probe : RecordingProbe | None = None,
        pretrained : bool = True,
        inp_shape : Tuple[int, ...] = (1, 3, 224, 224),
        device : str | torch.device = 'cuda',
    ) -> None:
        '''
        Initialize a subject represented by an artificial neural
        network capable of a visual task.

        :param network_name: Nme of the visual architecture to use
            for the subject. This should be one of the supported
            torchvision models (see torchvision.models for a list)
        :type network_name: string
        :param record_probe: Optional recording probe to attach to
            the network to record its activations when exposed to
            visual stimuli. (Default: None)
        :type record_probe: RecordingProbe or None
        :param pretrained: Flag to signal whether network should
            be initialized as pretrained (usually on ImageNet)
            (Default: True)
        :type pretrained: bool
        :param inp_shape: Shape of input tensor to the network,
            usual semantic is [B, C, H, W]. (Default: (1, 3, 224, 224))
        :type inp_shape: Tuple of ints
        :param device: Torch device where to host the module
            (Default: cuda)
        :type device: string or torch.device
        '''
        super().__init__()
        
        # Load the torch model via its name from the torchvision hub
        self._weights = models.get_model_weights(network_name) if pretrained else None
        self._network = models.get_model(network_name, weight=self._weights).to(device)

        # Attach NamingProbe to the network to properly assign name
        # to each layer so to get it ready for recording
        _name_hooks = self.register(NamingProbe())

        # Expose the network to a fake input to trigger the hooks
        mock_inp = torch.zeros(inp_shape, device=self.device)
        with torch.no_grad():
            _ = self._network(mock_inp)

        for hook in _name_hooks: hook.remove()

        # If provided, attach the recording probe to the network
        self._rec_probe = record_probe
        if self._rec_probe: self.register(self._rec_probe)

    @torch.no_grad()
    def forward(
        self,
        inp : Tensor,
        auto_clean : bool = True
    ) -> SubjectState:
        '''
        Expose NetworkSubject to a (visual input) and return the
        measured (hidden) activations. If no recording probe was
        registered to the subject this function raises an error

        :param inp: Input tensor (of expected shape [B, C, H, W])
        :type inp: (Torch) Tensor
        :param auto_clean: Flag to trigger RecordingProbe clean
            method after recording has taken place (Default: True)
        :type auto_clean: bool
        :returns: The measured subject state
        :rtype: SubjectState
        '''
        warn_msg = '''
                    Calling subject forward while no recording probe has been registered.
                    Please attach a recording probe via the `register` method of the
                    NetworkSubject class. 
                    '''
        assert self._rec_probe is not None, warn_msg     

        _ = self._network(inp)
    
        out = self._rec_probe.features
    
        if auto_clean: self._rec_probe.clean()
        
        return out

    def register(self, probe : SilicoProbe) -> List[RemovableHandle]:
        '''
        Attach a given SilicoProbe to the NetworkSubject by registering
        it as a forward_hook to the underlying model layers.

        :param probe: Probe to attach to the NetworkSubject
        :type probe: SilicoProbe (one of its concrete implementations)
        :returns: List of torch handles to release the attached hooks
        :rtype: List of RemovableHandle
        '''
        return [layer.register_forward_hook(probe) for layer in unpack(self._network)]
    
    @property
    def device(self) -> torch.device:
        return next(self._network.parameters()).device
    
    @property
    def layer_names(self) -> List[str]:
        '''
        Return layers names in the network architecture.
        
        :return: List of layers names.
        :rtype: List[str]
        '''
        return [layer.name for layer in unpack(self.network)]
    
    def get_layer(self, layer_name: str) -> nn.Module | None:
        '''
        Return the network layer matching the name in input.
        NOTE The layer is expected to have attribute "name" which
            is its identifier in layer indexing
        
        :param layer_name: Layer name in the architecture.
        :type layer_name: str
        :return: Network layer.
        :rtype: nn.Module
        '''
        for layer in unpack(self.network):
            if layer_name == layer.name: return layer

        return None