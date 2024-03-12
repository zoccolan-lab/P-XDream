from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List
from typing import Tuple, cast
import numpy as np

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from torchvision.models import get_model
from torchvision.models import get_model_weights

from .utils.model import Message, RecordingUnit
from .utils.model import Stimuli
from .utils.model import SubjectState
from .probe import SetterProbe
from .probe import RecordingProbe
from .probe import SilicoProbe
from .utils.model import InputLayer
from .utils.misc import default
from .utils.misc import device
from .utils.misc import unpack
from .utils.misc import replace_inplace

class Subject(ABC):
    '''
        Abstract class representing a subject (animal or network)
        tasked with a visual stimuli and generating a set of
        activations for multiple layers.
        
    '''
    
    # @abstractmethod
    # def foo(self):
    #     # TODO discuss what is a "Subject"
    #     pass
    
    
class InSilicoSubject(Subject):

    def __init__(self) -> None:
        super().__init__()
        self._states: List[SubjectState] = []
    
    @abstractmethod
    def __call__(
        self,
        data : Tuple[Stimuli, Message]
    ) -> Tuple[SubjectState, Message]:
        
        raise NotImplementedError("Cannot instantiate a InSilicoSubject")
    
    @property
    @abstractmethod
    def target(self) -> Dict[str, RecordingUnit]:
        pass
    
    @property
    def states_history(self) -> SubjectState:
        ''' 
        Returns the history of subject states as a single
        subject state with stacked activations
        '''
        if not self._states:
            raise ValueError("No state produced yet. ") 
        
        keys = self._states[0].keys()
        
        return {
            key: np.stack([state[key] for state in self._states])
            for key in keys
    }
    
    
    
class NetworkSubject(InSilicoSubject, nn.Module):
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
        device : str | torch.device = device,
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
        
        self._name = network_name
        self._inp_shape = inp_shape
        
        # Load the torch model via its name from the torchvision hub
        self._weights = get_model_weights(self._name).DEFAULT if pretrained else None # type: ignore
        self._network = nn.Sequential(
            InputLayer(),
            get_model(self._name, weights=self._weights)
        ).to(device)
        
        # NOTE: Here we make sure no inplace operations are used in the network
        #       to avoid weird behaviors (e.g. if a backward hook is attached
        #       to the network) at the cost of small memory increase
        replace_inplace(self._network)
        
        self._probes : Dict[SilicoProbe, List[RemovableHandle]] = defaultdict(list)

        # Attach NamingProbe to the network to properly assign name
        # to each layer so to get it ready for recording
        setter_probe = SetterProbe()
        self.register_forward(setter_probe)

        # Expose the network to a fake input to trigger the hooks
        mock_inp = torch.zeros(inp_shape, device=self.device)
        with torch.no_grad():
            _ = self._network(mock_inp)

        self.remove(setter_probe)

        # If provided, attach the recording probe to the network
        if record_probe: 
            self._target = record_probe.target
            self.register(record_probe)

    @property
    def target(self) -> Dict[str, RecordingUnit]:
        if hasattr(self, '_target'):
            return self._target
        return dict()
        
    def __str__(self) -> str:

        sep = ', '
        all = 'all'
        recording_targets = f"{sep.join([f'{k}: {len(v) if v else all} units' for k, v in self._target.items()])}"
        
        return f'NetworkSubject[name: {self._name}, in-shape: {self._inp_shape}, n-layers: {len(self.layer_names)},'\
               f' n-probes: {len(self._probes)}, recording: ({recording_targets})]'
        
    def __call__(
        self,
        data : Tuple[Stimuli, Message],
        probe : RecordingProbe | None = None,
        auto_clean : bool = True,
        raise_no_probe : bool = True,
    ) -> Tuple[SubjectState, Message]:
        warn_msg = \
            '''
            Calling subject forward while no recording probe has been registered.
            This will result in subject forward output to have an empty SubjectState
            which may lead in downstream failure. Please be mindful or the consequences
            or register a recording probe via the `register` method of the NetworkSubject
            class. 
            '''

        # TODO: This return only the first RecorderProbe, what if
        #       more than one were registered?        
        probe = default(probe, self.recorder)
        
        if not probe and probe not in self._probes:
            if raise_no_probe: assert False, warn_msg
            else: print(warn_msg)

        state, msg =  self.forward(
            data=data,
            probe=probe,
            auto_clean=auto_clean
        )

        self._states.append(state)

        return state, msg

    # @torch.no_grad()
    def forward(
        self,
        data : Tuple[Stimuli, Message],
        probe : RecordingProbe | None = None,
        auto_clean : bool = True,
    ) -> Tuple[SubjectState, Message]:
        '''
        Expose NetworkSubject to a (visual input) and return the
        measured (hidden) activations. If no recording probe was
        registered to the subject this function raises an error

        :param data: Input tensor (of expected shape [B, C, H, W])
        :type data: Tuple[Stimuli, Message]
        :param probe:
        :type probe: RecordingProbe
        :param auto_clean: Flag to trigger RecordingProbe clean
            method after recording has taken place (Default: True)
        :type auto_clean: bool
        :returns: The measured subject state
        :rtype: SubjectState
        '''

        stimuli, msg = data
        
        pipe = self._weights.transforms() if self._weights else lambda x : x
        stimuli = pipe(stimuli)

        _ = self._network(stimuli)
    
        out = probe.features if probe else {}  
    
        if probe and auto_clean: probe.clean()
        
        return out, msg
    
    def register(self, probe : SilicoProbe) -> Tuple[List[RemovableHandle], List[RemovableHandle]]:
        fw_handles = self.register_forward (probe)
        bw_handles = self.register_backward(probe)

        return fw_handles, bw_handles

    def register_forward(self, probe : SilicoProbe) -> List[RemovableHandle]:
        '''
        Attach a given SilicoProbe to the NetworkSubject by registering
        it as a forward_hook to the underlying model layers.

        :param probe: Probe to attach to the NetworkSubject
        :type probe: SilicoProbe (one of its concrete implementations)
        :returns: List of torch handles to release the attached hooks
        :rtype: List of RemovableHandle
        '''

        handles = [layer.register_forward_hook(probe.forward) for layer in unpack(self._network)]
        self._probes[probe] = handles
        
        return handles
    
    def register_backward(self, probe : SilicoProbe) -> List[RemovableHandle]:
        '''
        Attach a given SilicoProbe to the NetworkSubject by registering
        it as a forward_hook to the underlying model layers.

        :param probe: Probe to attach to the NetworkSubject
        :type probe: SilicoProbe (one of its concrete implementations)
        :returns: List of torch handles to release the attached hooks
        :rtype: List of RemovableHandle
        '''

        handles = [layer.register_full_backward_hook(probe.backward) for layer in unpack(self._network)]
        self._probes[probe] = handles
        
        return handles
    
    def remove(self, probe : SilicoProbe) -> None:
        '''
        Remove the hooks associated to the provided probe
        '''
        probe.clean()
        handles = self._probes.pop(probe)
        for hook in handles: hook.remove()

    def remove_all(self) -> None:
        for handles in self._probes.values():
            for hook in handles: hook.remove()

        self._probes = defaultdict(list) 

    def get_layer(self, layer_name: str) -> nn.Module | None:
        '''
        Return the network layer matching the name in input.
        NOTE: The layer is expected to have attribute "name" which
              is its identifier in layer indexing
        
        :param layer_name: Layer name in the architecture.
        :type layer_name: str
        :return: Network layer.
        :rtype: nn.Module
        '''
        for layer in unpack(self._network):
            if layer_name == layer.name: return layer

        return None
    
    @property
    def recorder(self) -> RecordingProbe | None:
        for probe in self._probes:
            if isinstance(probe, RecordingProbe):
                return probe
            
        return None
    
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
        return [layer.name for layer in unpack(self._network)]
    
    
    @property
    def layer_shapes(self) -> List[tuple[int, ...]]:
        '''
        Return layers shapes in the network architecture.
        
        :return: List of layers shapes.
        :rtype: List[tuple[int, ...]]
        '''
        return [layer.shape for layer in unpack(self._network)]
    
    @property
    def layer_info(self) -> dict[str, tuple[int, ...]]:
        '''
        Return layers shapes in the network architecture.
        
        :return: Dictionary containing layers name and shapes.
        :rtype: dict[str, tuple[int, ...]]
        '''
        names = self.layer_names
        shapes = self.layer_shapes
        
        return {k : v for k, v in zip(names, shapes)}