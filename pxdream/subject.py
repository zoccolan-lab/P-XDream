'''
This file contains the definition of the Subject class and its concrete implementation.
A Subject is an entity that can be exposed to visual stimuli and return a set of activations for multiple layers.

The class architecture in thought to be extended both to in-silico subjects (e.g. artificial networks) and in-vivo ones (e.g. animals).
'''

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
import torch.nn.functional as F
from torchvision.models import get_model, get_model_weights

from .utils.logger import Logger, SilentLogger
from .utils.probe import SetterProbe, SilicoProbe,RecordingProbe
from .utils.misc import InputLayer, default, device, unpack, replace_inplace
from .utils.types import RecordingUnits, Stimuli, States
# from robustness.datasets import ImageNet
# from robustness.model_utils import make_and_restore_model


class Subject(ABC):
    '''
    Abstract class representing a subject (animal or network)
    tasked with a visual stimuli and generating a set of
    activations for multiple layers.
    
    NOTE: This class will serve as a common interface between in-silico subjects
        and in-vivo one when the framework will be extended to support surgery.
    '''

    pass


class InSilicoSubject(Subject):
    '''
    This class represents a generic in-silico subject, i.e. an artificial
    network involved in a visual task experiment as an artificial counterpart
    of an animal. 
    
    The class defines the interface for the subject to be exposed to visual stimuli 
    and return a set of activations for multiple layers.
    '''

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(
            self,
            stimuli: Stimuli
    ) -> States:
        '''
        Abstract method to expose the subject to a visual input and return
        the measured activations.

        :param stimuli: Input batched tensor of stimuli.
        :type stimuli: Stimuli
        :return: The measured subject state.
        :rtype: State
        '''

        raise NotImplementedError("Cannot instantiate an InSilicoSubject")

    @property
    @abstractmethod
    def target(self) -> Dict[str, RecordingUnits]:
        '''
        Returns the units to record for each layer in the network.

        :return: Dictionary containing the units to record for each layer.
        :rtype: Dict[str, RecordingUnit]
        '''
        pass


class TorchNetworkSubject(InSilicoSubject, nn.Module):
    '''
    Class representing an artificial network involved in
    a visual task experiment as an artificial counterpart 
    of an animal that uses torch modules as building blocks.
    
    The class serves of an NamingProbe that provides an unique 
    identifier to each layer (via a NamingProbe) that can be 
    used for recording.
    '''

    def __init__(
        self,
        network_name: str,
        record_probe: RecordingProbe | None = None,
        model_weights : PathLike | Literal['pretrained'] = 'pretrained', 
        inp_shape: Tuple[int, ...] = (1, 3, 224, 224),
        device: str | torch.device = device,
    ) -> None:
        '''
        Initialize a subject represented by an artificial neural
        network capable of a visual task.
        
        1) It loads the network architecture from the torchvision hub
        based on the input architecture name and initializes it
        with pretrained weights if requested.
        
        2) It runs a SetterProbe to assign a unique identifier to each
        layer in the network that can be used for recording.
        
        :param network_name: Name of the visual architecture to use
            for the subject. This should be one of the supported
            torchvision models (see torchvision.models for a list)
        :type network_name: string
        :param record_probe: Optional recording probe to attach to the network to record 
            its activations when exposed to visual stimuli. Default to None
            indicating that no recording will take place.
        :type record_probe: RecordingProbe | None
        :param pretrained: Flag to signal whether network should be initialized as pretrained,
            default to True.
        :type pretrained: bool
        :param inp_shape: Shape of input tensor to the network, usual semantic is [B, C, H, W].
            Defaults to (1, 3, 224, 224).
        :type inp_shape: Tuple[int, ...]
        :param device: Torch device where to host the module, defaults to cuda.
        :type device: string | torch.device
        '''

        super().__init__()

        # Initialize the network with the provided name and input shape
        self._name = network_name
        self._inp_shape = inp_shape
        self.robust = '_r' if custom_weights_path else ''
        
        # 1) LOAD NETWORK ARCHITECTURE
        
        if custom_weights_path != '':
            self._weights = torch.load(custom_weights_path)
        else:
            # Load the torch model via its name from the torchvision hub if pretrained
            # otherwise initialize it with random weights (weights=None).
            self._weights = get_model_weights(self._name).DEFAULT if pretrained else None  # type: ignore

        self._network = nn.Sequential(
            InputLayer(),
            get_model(self._name, weights=self._weights)
        ).to(device)

        # NOTE: Here we make sure no inplace operations are used in the network
        #       to avoid weird behaviors (e.g. if a backward hook is attached
        #       to the network) at the cost of small memory increase
        replace_inplace(self._network)

        # 2) ATTACH SETTER PROBE

        # Initialize the probes dictionary to store the attached probes
        self._probes: Dict[SilicoProbe, List[RemovableHandle]] = defaultdict(list)

        # Attach a SetterProbe to the network to properly 
        # assign a unique identifier to each layer and
        # retrieve it's input shape
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
            self.register_forward(record_probe)
            


    # --- STRING REPRESENTATION ---

    def __str__(self) -> str:
        '''
        Return the string representation of the NetworkSubject.
        '''

        sep = ', '
        all = 'all'  # NOTE: RecordingUnit equal to None indicates to record all units

        # Create a string with the number of units to record for each layer
        recording_targets = f"{sep.join([f'{k}: {len(v) if v else all} units' for k, v in self._target.items()])}"

        return  f'NetworkSubject[name: {self.name}, ' \
                f'in-shape: {self.in_shape}, ' \
                f'n-layers: {len(self.layer_names)}, ' \
                f'n-probes: {len(self._probes)}, ' \
                f'recording: ({recording_targets})]'

    def __repr__(self) -> str:
        return str(self)

    ''' Return the string representation of the NetworkSubject.'''

    # --- PROPERTIES ---

    @property
    def target(self) -> Dict[str, RecordingUnits]:
        '''
        Return the units to record for each layer in the network.

        :return: Dictionary containing the units to record for each layer.
        :rtype: Dict[str, RecordingUnit]
        '''

        # If a recording probe was attached to the network
        # return the target units for each layer
        if hasattr(self, '_target'):
            return self._target

        # Otherwise return an empty dictionary
        return dict()

    @property
    def in_shape(self) -> Tuple[int, ...]:
        return self._inp_shape

    ''' Return the input shape of the network. '''

    @property
    def recorder(self) -> RecordingProbe | None:
        '''
        Return the recording probe attached to the network if any.

        :return: Recording probe attached to the network. 
            In case no probe was attached, return None.
        :rtype: RecordingProbe | None
        '''

        # Return the first recording probe attached to the network
        for probe in self._probes:
            if isinstance(probe, RecordingProbe):
                return probe

        # If no recording probe was attached, return None
        return None
    
    @property
    def name(self) -> str: return self._name
    ''' Return the name of the network architecture. '''

    @property
    def device(self) -> torch.device: return next(self._network.parameters()).device
    ''' Return the device where the network is hosted. '''

    @property
    def layer_names(self) -> List[str]: return [layer.name for layer in unpack(self._network)]
    ''' Return layers names in the network architecture. '''

    @property
    def layer_shapes(self) -> List[Tuple[int, ...]]: return [layer.shape for layer in unpack(self._network)]
    ''' Return layers shapes in the network architecture. '''

        

    @property
    def layer_info(self) -> Dict[str, Tuple[int, ...]]:
        '''
        Return layers names and shapes in the network architecture.
        
        :return: Dictionary containing layers name and shapes.
        :rtype: dict[str, tuple[int, ...]]
        '''

        names = self.layer_names
        shapes = self.layer_shapes

        return {k: v for k, v in zip(names, shapes)}

    # --- RECORDING ---

    def __call__(
            self,
            stimuli        : Stimuli,
            probe          : RecordingProbe | None = None,
            auto_clean     : bool = True,
            raise_no_probe : bool = True,
            with_grad      : bool = False,
            logger         : Logger = SilentLogger()
    ) -> States:
        '''
        Expose the network to the visual input and return the measured activations.

        The function is a wrapper of the forward method implementing the core of
        implementation, but handles outside image preprocessing.

        :param stimuli: Optional recording probe to attach to the network to record.
            If not given the first registered probe will be used.
            In the case there are no probes registered, a warning message is logged.
        :type stimuli: Stimuli
        :param probe: Optional recording probe to attach to the network to record,
            if not provided the first registered probe will be used.
        :type probe: RecordingProbe | None
        :param auto_clean: Flag to trigger RecordingProbe clean method after recording
        :type auto_clean: bool
        :param raise_no_probe: Flag to raise an error if no probe was registered.
        :type raise_no_probe: bool
        :param with_grad: Flag to enable gradient computation during forward pass
        :type with_grad: bool
        :param logger: Logger to use for logging messages, defaults to SilentLogger.
        :type logger: Logger
        '''

        # TODO: This return only the first RecorderProbe, what if
        # TODO: more than one were registered?
        # TODO: @Paolo

        # If no probe was provided, use the first registered probe
        probe = default(probe, self.recorder)

        # Check no probe for recording
        if not probe and probe not in self._probes:

            warn_msg = \
                '''
                Calling subject forward while no recording probe has been registered.
                This will result in subject forward output to have an empty State
                which may lead in downstream failure. Please be mindful or the consequences
                or register a recording probe via the `register` method of the NetworkSubject class. 
                '''

            # If the flag is set to raise an error, raise it
            if raise_no_probe:
                assert False, warn_msg

            # Otherwise print a warning message
            else:
                logger.warn(warn_msg)

        # Apply preprocessing associated to pretrained weights
        # NOTE: `self.weights` is None in the case of random initialization
        #       and corresponds to no transformation
        preprocessing = self._weights.transforms() if self._weights and not(self.robust) else lambda x: x
        prep_stimuli = preprocessing(stimuli)

        # Expose the network to the visual input and return the measured activations
        state = self.forward(
            stimuli=prep_stimuli,
            probe=probe,
            auto_clean=auto_clean,
            with_grad=with_grad
        )
        #get the input as 224x224x3 image with values in range (0,1) as in Gaziv et al. 2023
        state['00_input_01'] = F.interpolate(stimuli, size=(224, 224), mode='bilinear', align_corners=False).view(stimuli.shape[0],-1).cpu().numpy().astype('float32')
        
        return state

    def forward(
            self,
            stimuli: Stimuli,
            probe: RecordingProbe | None = None,
            auto_clean: bool = True,
            with_grad: bool = False
    ) -> States:
        '''
        Forward pass of the network with the given stimuli.

        :param stimuli: Input batched tensor of stimuli.
        :type stimuli: Stimuli
        :param probe: Optional recording probe to attach to the network to record.
        :type probe: RecordingProbe | None, optional
        :param auto_clean: Flag to trigger RecordingProbe clean method after recording
            that will remove the recorded activations.
        :type auto_clean: bool, optional
        :param with_grad: Flag to enable gradient computation during forward pass
        :type with_grad: bool, optional
        :return: The measured subject state.
        :rtype: State
        '''

        # Expose the network to the visual input and return the measured activations
        # Compute the forward pass with or without gradient computation based on the input flag
        if with_grad:
            _ = self._network(stimuli)
        else:
            with torch.no_grad():
                _ = self._network(stimuli)

        states = probe.features if probe else {}

        if probe and auto_clean: probe.clean()

        return states

    # --- PROBES ---

    def register(self, probe: SilicoProbe) -> Tuple[List[RemovableHandle], List[RemovableHandle]]:
        '''
        Attach a given SilicoProbe to the NetworkSubject by registering

        :param probe: Probe to attach to the NetworkSubject.
        :type probe: SilicoProbe
        :return: Tuple of torch handles to release the attached hooks.
        :rtype: Tuple[List[RemovableHandle], List[RemovableHandle]]
        '''

        # Register the probe as both forward and backward hook
        fw_handles = self.register_forward(probe)
        bw_handles = self.register_backward(probe)

        return fw_handles, bw_handles

    def register_forward(self, probe: SilicoProbe) -> List[RemovableHandle]:
        '''
        Attach a given SilicoProbe to the NetworkSubject by registering
        it as a forward_hook to the underlying model layers.

        :param probe: Probe to attach to the NetworkSubject
        :type probe: SilicoProbe (one of its concrete implementations)
        :returns: List of torch handles to release the attached hooks
        :rtype: List[RemovableHandle]
        '''

        # Register the probe as a forward hook to the network layers
        handles = [
            layer.register_forward_hook(probe.forward)
            for layer in unpack(self._network)
        ]

        # Store the handles in the probes dictionary
        self._probes[probe] = handles

        return handles

    def register_backward(self, probe: SilicoProbe) -> List[RemovableHandle]:
        '''
        Attach a given SilicoProbe to the NetworkSubject by registering
        it as a forward_hook to the underlying model layers.

        :param probe: Probe to attach to the NetworkSubject
        :type probe: SilicoProbe (one of its concrete implementations)
        :returns: List of torch handles to release the attached hooks
        :rtype: List of RemovableHandle
        '''

        # Register the probe as a backward hook to the network layers
        handles = [
            layer.register_full_backward_hook(probe.backward)
            for layer in unpack(self._network)
        ]

        # Store the handles in the probes dictionary
        self._probes[probe] = handles

        return handles

    def clean(self) -> None:
        ''' Clean subject states and probe content '''

        # Clean the states and probes content
        self._states: List[States] = []

        # Clean the recorder content
        if self.recorder is not None:
            self.recorder.clean()

    def remove(self, probe: SilicoProbe) -> None:
        ''' Remove the hooks associated to the provided probe '''

        # Remove the hooks associated to the provided probe
        probe.clean()

        # Remove the probe from the probes dictionary
        handles = self._probes.pop(probe)

        # Remove the hooks from the network layers
        for hook in handles: hook.remove()

    def remove_all(self) -> None:
        ''' Remove all the hooks associated to the probes '''

        # Remove all the hooks associated to the probes
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
        :return: Network layer if found, None otherwise.
        :rtype: nn.Module
        '''

        # Return the layer matching the input name
        for layer in unpack(self._network):
            if layer_name == layer.name: return layer
