'''
TODO Document with @Paolo and @Lorenzo
'''

from abc import ABC, abstractmethod
from itertools import product
from collections import defaultdict, OrderedDict
from math import prod
from typing import Dict, Tuple, List, Any, Literal
from uuid import uuid4

from einops import rearrange, reduce
import numpy as np
from numpy.typing import DTypeLike, NDArray
import torch.nn as nn
from torch import Tensor

from .misc import InputLayer, default, fit_bbox
from .types import RFBox, RecordingUnits, States

# TODO Document together

class SilicoProbe(ABC):
    '''
    This is an abstract probe that can be used with an artificial neural network. 
    
    It is designed to work with a torch hook, typically a forward hook, 
    implemented through the __call__ function. 
    
    The probe is responsible for performing a specific task,
    which can involve returning a value or causing side effects.
    '''

    def __init__(self) -> None:
        ''' Initialize the probe with a unique identifier.'''
        
        self.unique_id = uuid4()
        
    # --- HASHABLE METHODS ---

    def __hash__(self) -> int: return hash(self.unique_id)
    ''' Return the hash of the unique identifier.'''
    
    def __eq__(self, other : 'SilicoProbe') -> bool: return self.unique_id == other.unique_id
    ''' Check if two probes are equal by comparing their unique identifiers. '''
    
    # --- HOOKS ---
    
    @abstractmethod
    def forward(
        self,
        module : nn.Module,
        inp    : Tuple[Tensor, ...],
        out    : Tensor,
    ) -> Tensor | None:
        '''
        Abstract implementation of PyTorch forward hook. Each Probe subclass is expected 
        to provide the specific implementation to accomplish its given task.
        
        
        :param module: The calling torch Module who raised the hook callback.
        :type module: torch.nn.Module
        :param inp: The input to the calling module.
        :type inp: Tuple[Tensor, ...]
        :param out: The computed calling module output (callback is raised after at the end of the forward_step).
        :type out: Tensor
        :returns: The output of the forward hook. If None is returned, the hook will not have any effect.
        :rtype: Tensor | None 
        '''
        pass

    def backward(
        self,
        module : nn.Module,
        grad_inp : Any, # TODO: Cannot find the appropriate type to put here
        grad_out : Any, # TODO: Cannot find the appropriate type to put here
    ) -> Tensor | None:
        '''
        Abstract implementation of PyTorch backward hook. Each Probe subclass is expected
        to provide the specific implementation to accomplish its given task.
        
        By default, this function raises a NotImplementedError since not all probes
        are expected to have a backward hook.
        '''
        
        raise NotImplementedError(f'Probe {self} does not support backward hook')

    @abstractmethod
    def clean(self) -> None: pass
    '''
    Abstract method to clean the probe from any stored data. This method is
    expected to be called at the end of a forward pass to reset the probe
    to its initial state.
    '''

class SetterProbe(SilicoProbe):
    '''
    Simple probe whose task is to attach to a given torch Module
    a unique identifier to each of its sub-modules (layers) and
    store the output shape of each of these layers.
    '''

    def __init__(self, exclude : List[str] | None = None) -> None:
        '''
        Construct a SetterProbe by specifying the name of the
        attribute that the probe attaches to each sub-module of
        the target torch.nn.Module as its unique identifier.
        '''
        
        super().__init__()
        
        # Define the attribute names to attach to each module
        self.name_attr  = 'name'
        self.shape_attr = 'shape'
        self.exclude    = default(exclude, [])
        
        # Initialize layer depth and occurrence dictionary
        # by triggering the clean method
        self.clean()
        

    def forward(
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
        if hasattr(module, self.name_attr): return
        if hasattr(module, self.shape_attr): return
        
        # Save the current layer name and output shape
        name = module._get_name().lower()
        curr_shape = out.detach().cpu().numpy().shape
        
        if name in self.exclude:
            # Attach the unique identifier to the (sub-)module
            setattr(module, self.name_attr,  'excluded')
            setattr(module, self.shape_attr, None)    
            return
        
        # Update the depth and occurrence of this layer type
        self._depth       += 1
        self._occur[name] += 1
        
        # Build a unique identifier for this module based
        # on module name, depth and layer type occurrence
        depth = str(self._depth      ).zfill(2)
        occur = str(self._occur[name]).zfill(2)
        identifier = f'{depth}_{name}_{occur}'
        
        # Attach the unique identifier to the (sub-)module
        setattr(module, self.name_attr,  identifier)
        setattr(module, self.shape_attr, curr_shape)

    def clean(self) -> None:
        '''
        Reset the depth and occurrence dictionary
        '''
        
        # Initialize the layer depth 
        self._depth = -1
        
        # Initialize the dictionary storing the type
        # has been encountered at a given depth
        self._occur = defaultdict(lambda : 0)

class InfoProbe(SilicoProbe):
    '''
    This is a simple probe that retrieves useful information for a torch Module. 
    It measures the shape of target layers and the (generalized) receptive fields of selected target units.
    
    The "generalized" receptive field can be defined as either the input level, 
    which matches the standard definition in neuroscience for a receptive field, 
    or chosen at a preceding layer, in which case it returns the set of units from 
    which the target receives information.
    '''
    
    
    # Define the types of layers that can be encountered
    # in catgories based on their behavior
    
    MeanLike = (
        nn.AvgPool2d, 
    )
    DownLike = (
        nn.ConvTranspose2d, 
    )
    ConvLike = (
        nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.Conv3d, nn.MaxPool3d
    )
    
    # NOTE: List of PassLike is taken from:
    #       https://github.com/Fangyh09/pytorch-receptive-field 
    PassLike = (
        nn.AdaptiveAvgPool2d, nn.BatchNorm2d,    nn.Linear,        nn.ReLU,      nn.LeakyReLU,
        nn.ELU,               nn.Hardshrink,     nn.Hardsigmoid,   nn.Hardtanh,  nn.LogSigmoid, nn.PReLU,
        nn.ReLU6,             nn.RReLU, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid,   nn.SiLU,       nn.Mish,
        nn.Softplus,          nn.Softshrink,     nn.Softsign,      nn.Tanh,      nn.Tanhshrink, nn.Threshold,
        nn.Dropout,           nn.Dropout1d,      nn.Dropout2d,     nn.Dropout3d, InputLayer
    )
    
    def __init__(
        self,
        inp_shape : Tuple[int, ...],
        rf_method : Literal['forward', 'backward'] = 'forward',
        forward_target : None | Dict[str, RecordingUnits] = None,
        backward_target: None | Dict[str, Dict[str, RecordingUnits]] = None,
    ) -> None:
        '''
        Construct a InfoProbe by specifying the input shape of the network,
        the method to compute the receptive field and the target units for
        which to compute the receptive field.

        :param inp_shape: The shape of the input to the network-
        :type inp_shape: Tuple[int, ...]
        :param rf_method: The method to use to compute the receptive field, either 'forward' or 'backward'.
            Defaults to 'forward'.
        :type rf_method: Literal['forward', 'backward']
        :param forward_target: The target units for which to compute the receptive field. 
            If None, TODO
        :type forward_target: None | Dict[str, RecordingUnit], optional
        :param backward_target: The target units for which to compute the receptive field.
            If None, TODO
        :type backward_target: None | Dict[str, Dict[str, RecordingUnit]], optional
        '''
        
        super().__init__()
        
        # Store the input shape of the network
        self.rf_method = rf_method
        self._f_target = forward_target
        self._b_target = backward_target
        
        # Initialize data structures to store the receptive field information
        self._rf_dict: Dict[str, List]                             = defaultdict(list)     # TODO Doc @Paolo
        self._output : Dict[Tuple[str, str], Tensor]               = {}                    # TODO Doc @Paolo 
        self._shapes : Dict[str, Tuple[int, ...]]                  = {'input' : inp_shape} # TODO Doc @Paolo
        self._ingrad : Dict[Tuple[str, str], List[NDArray | None]] = defaultdict(list)     # TODO Doc @Paolo
        
        # TODO Doc @Paolo
        self._rf_par : Dict[str, Dict[str, float]] = OrderedDict(
            input={
                'jump' : 1,
                'size' : 1,
                'start' : 0.5,
            }
        )
        
    def forward(
        self,
        module: nn.Module,
        inp: Tuple[Tensor],
        out: Tensor
    ) -> None:
        '''
        Forward hook used to gather information about the receptive field.

        :param module: Torch module the hook is called.
        :type module: nn.Module
        :param inp: Unused, but expected by the overriding.
        :type inp: Tuple[Tensor]
        :param out: The output of the module.
        :type out: Tensor
        '''
        
        # Check if the module has a name and shape attribute
        if not hasattr(module, 'name'):
            raise AttributeError(f'Encounter module {module} with unregistered name.')
        
        if not hasattr(module, 'shape'):
            raise AttributeError(f'Encounter module {module} with unregistered shape.')
        
        # Save the current layer name and output shape
        curr_name  = module.name
        curr_shape = module.shape
        
        # Update the shape dictionary with the current layer
        self._shapes[curr_name] = curr_shape
        
        # NOTE: We check whether this layer output is needed for
        #       a backward pass because it was requested by the
        #       backward hook functionality        
        if self._b_target:
            
            for ref, targets in self._b_target.items():
                
                try:
                    targ_idx = targets[curr_name]
                    targ_act = out if targ_idx is None else out[(slice(None), *targ_idx)]
                    
                    # Rearrange target activation to have common shape
                    # NOTE: We expect batch dimension to have singleton shape
                    self._output[(ref, curr_name)] = reduce(targ_act, 'b ... -> (...)', 'mean')

                # No worries if current layer is not among the
                # backward targets, just pass
                except KeyError: pass
        
        # Collect parameters needed for the RF computation
        # Here we get the last-inserted (hence previous) key-val
        # pair in the rf_par dictionary, which corresponds to the
        # parameters of the previous layer this hook was call by
        p_val = next(reversed(self._rf_par.values()))

        if isinstance(module, self.ConvLike):
            
            s, p, k = module.stride, module.padding, module.kernel_size
            
            d = 1 if isinstance(module, self.MeanLike) else module.dilation
            
            s, p, k, d = map(self._sanitize, (s, p, k, d))
            
            # Update the current layer parameter for RF computation
            self._rf_par[curr_name] = {
                'jump' : p_val['jump'] * s,
                'size' : p_val['size'] + ((k - 1)     * d) * p_val['jump'],
                'start': p_val['start']+ ((k - 1) / 2 - p) * p_val['start'],
            }  
        
        elif isinstance(module, self.PassLike): self._rf_par[curr_name] = p_val.copy()
        elif isinstance(module, self.DownLike): self._rf_par[curr_name] = {k : 0 for k in p_val}
        else : raise TypeError(f'Encountered layer of unknown type: {module}')

    def backward(
        self,
        module : nn.Module,
        grad_inp : Tuple[Tensor | None, ...],
        grad_out : Tuple[Tensor | None, ...],
    ) -> None:
        '''
        Backward hook used to gather information about the receptive
        field of a given set of units either at the input level (i.e.
        the standard definition of a receptive field) or at a given
        intermediate level (i.e. generalized receptive field).

        :param module: The calling torch module layer
        :type module: torch.nn.Module
        :param grad_inp: The gradient of model parameters with respect
            to the received input. Entries in grad_inp will be None
            for all non-Tensor arguments provided as layer input
        :type grad_inp: Either a tensor or a Tuple of Tensor or None.
        :param grad_out: The gradient of model parameters with respect
            to the received input. Entries in grad_inp will be None
            for all non-Tensor arguments
        :type grad_out: Either a tensor or a Tuple of Tensor or None.
        '''
        
        # Check if the module has a name attribute
        if not hasattr(module, 'name'):
            raise AttributeError(f'Encounter module {module} with unregistered name.')
        
        curr = module.name
        
        if curr not in self._b_target: return
        
        if isinstance(grad_out, tuple): grad, *_ = grad_out
        
        grad = grad.detach().abs().cpu().numpy() if grad is not None else None
        
        if grad is None: return
        
        grad_mean = np.mean(grad, axis = 0)
        
        if grad_mean.ndim == 1:
            
            grad_mean = np.expand_dims(grad_mean, axis=0)
            axes = tuple([-1])
            
        else:
            axes = tuple(-i for i in range(len(grad_mean.shape),0, -1))
        
        self._rf_dict[(curr, self.source)].append(fit_bbox(grad_mean, axes=axes))

        
    def _sanitize(self, var : int | str | Tuple) -> int:
        '''
        Sanitize the input variable to be an integer.

        :param var: The variable to sanitize.
        :type var: int | str | Tuple
        :return: The sanitized variable.
        :rtype: int
        '''
        
        # If a variable is a tuple, we expect it to be a range
        # specification and we return the first element
        if isinstance(var, (tuple, list)):
            assert  (len(var) == 2 and var[0] == var[1]) or\
                    (len(var) == 3 and var[0] == var[1] and var[1] == var[2])
            return var[0]
        
        #
        elif isinstance(var, str):
            raise ValueError(f'Cannot sanitize var of value: {var}')
        
        # If the variable is already an integer, we just return it
        else:             
            return var

    @property
    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self._shapes
    
    @property
    def rec_field(self) -> Dict[Tuple[str, str], List[RFBox]]:        
        match self.rf_method:
            case 'forward':
                err_msg = \
                '''
                Requested forward-flavored receptive field but not forward targets were specified.
                Please either construct probe with specified forward targets or explicitly set
                them via the `register_forward_target` dedicated method.
                '''
                assert self._f_target, err_msg
                return self._get_forward_rf(self._f_target)
            
            case 'backward':
                err_msg = \
                '''
                Requested backward-flavored receptive field but not forward targets were specified.
                Please either construct probe with specified forward targets or explicitly set
                them via the `register_backward_target` dedicated method.
                '''
                assert self._b_target, err_msg
                return self._get_backward_rf(self._b_target)
            case _: raise ValueError(f'Unknown requested receptive field method: {self.rf_method}')
        
    def _get_forward_rf(
        self,
        fw_target : Dict[str, RecordingUnits]
    ) -> Dict[Tuple[str, str], List[RFBox]]:
        fields : Dict[Tuple[str, str], List[RFBox]] = {}
        
        *_, w, h = self._shapes['input']
        
        for layer, target in fw_target.items():
            if layer not in self._rf_par:
                raise KeyError(f'No (forward) RF info is available for layer: {layer}.')
            
            f_par = self._rf_par[layer]
            shape = self._shapes[layer]
            num_units = len(target) if target is not None else prod(shape)
            
            if len(shape) < 3:
                fields[('00_input_01', layer)] = [(0, w, 0, h)] * num_units 
                continue
            
            rf_field = [(
                f_par['start'] + pos * f_par['jump'] - f_par['size'] / 2,
                f_par['start'] + pos * f_par['jump'] + f_par['size'] / 2)
                for *_, x, y in default(target, product(*[range(d) for d in shape]))
                for pos in (x, y)
            ]

            fields[('00_input_01', layer)] = [
                (max(0, int(x1)), min(w, int(x2)), max(0, int(y1)), min(h, int(y2)))
                for (x1, x2), (y1, y2) in zip(rf_field[::2], rf_field[1::2])
            ]
                        
        return fields
    
    def _get_backward_rf(
        self,
        bw_target : Dict[str, Dict[str, RecordingUnits]],
        act_scale : float = 1e2,
    ) -> Dict[Tuple[str, str], List[RFBox]]:
        # raise NotImplementedError()
        # TODO: Figure out how to implement this!
        I = len(self._output)
        for i, ((self.ref, self.source), targ_act) in enumerate(self._output.items()):
            for j, act in enumerate(targ_act[:-1]):                
                # * This is where we trigger the backward hook
                # NOTE: This call should populate the self._ingrad
                #       attribute of this class
                act.backward(retain_graph=True)
            targ_act[-1].backward(retain_graph=False) if i==(I-1) else targ_act[-1].backward(retain_graph=True)
                
        #TO DO: it seems that at each act.backward step all the gradients are backpropagated.
        #therefore we end up with a dict that, for each key, has a replica of its entry n times,
        #where n is equal to the number of layers onto which the mapping occurs.
        #I did a quick fix (i.e. taking the first n element of every value, where n is the number of units
        # considered) that seems to work for now
        return {k: v[:int(targ_act.size(0))] for k,v in self._rf_dict.items()}
        #return {
        #    k : [fit_bbox(np.mean(grad[0], axis = 0)) for grad in v]
        #    for k, v in self._ingrad.items()
        #}
    
    def clean(self) -> None:
        '''
        Remove all stored data from internal storage 
        '''
        self._output = {}
        self._shapes = {}
        self._ingrad = defaultdict(list)
        self._rf_par = OrderedDict(
            input={
                'jump' : 1,
                'size' : 1,
                'start' : 0.5,
            }
        )
        
class RecordingProbe(SilicoProbe):
    '''
    Basic probe to attach to an artificial network to
    record the activations of an arbitrary number of
    hidden units in arbitrary layers. 
    '''
    
    def __init__(
        self,
        target : Dict[str, RecordingUnits],
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
    def target(self) -> Dict[str, RecordingUnits]: return self._target
        
    @property
    def features(self) -> States:
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
        
    def forward(
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
        # TODO: Is this line actually necessary?
        targ_act = rearrange(targ_act.astype(self._format), 'b ... -> b (...)')
        
        # Register the network activations in probe data storage
        self._data[module.name].append(targ_act)

        
    def clean(self) -> None:
        '''
        Remove all stored activations from data storage 
        '''
        self._data = defaultdict(list)