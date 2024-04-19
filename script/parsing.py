from collections import defaultdict
from itertools import product, starmap
from typing import Dict, List, Tuple, Callable
import warnings

from zdream.utils.types import RecordingUnit, ScoringUnit
from zdream.utils.io_ import neurons_from_file

import numpy as np

# --- SCORING and AGGREGATE FUNCTION TEMPLATES ---

def parse_boolean_string(boolean_str: str) -> List[bool]:
    ''' Converts a boolean string of T and F characters in a boolean list'''

    def char_to_bool(ch: str) -> bool:
        match ch:
            case 'T': return True
            case 'F': return False
            case _: raise ValueError('Boolean string must contain only T and F symbols.')

    return[char_to_bool(ch=ch) for ch in boolean_str]

def parse_int_list(integer_list_str: str) -> List[int]:
    ''' Converts string of integers separated by comma to a list of integers'''
    
    if integer_list_str:
        return[int(n) for n in integer_list_str.split(',')]
    return []

def parse_signature(
    input_str : str,
    net_info: Dict[str, Tuple[int, ...]],
) -> Dict[str, float]:
    '''
    Converts and input string indicating the signature for the
    WeightedPairSimilarityScorer into the appropriate type
    (a dictionary mapping layer-name-string to floats).
    
    The format to specify the signature requires separating specification
    for different layers with comma:
        layer1=signature1, layer2=signature2, ..., layerN=signatureN
        
    Where signature specification is expected to be a string that
    can be parsed by float(signature).
    
    Example:
    signature = parse_signature(
        'conv2d_03=4.8, relu_08=-0.77, linear_03=+0.001' 
    )
    
    print(signature)
    > {
        'conv2d_03' : 4.8,
        'relu_08' : -0.77,
        'linear_03' : 0.001,
    }
    '''
    
    # Extract layer names from the net information 
    # for mapping layer-IDs to their name
    layer_names = list(net_info.keys())
    
    signature = {
        layer_names[int(k.strip())] : float(v)
        for k, v in [layer.split('=') for layer in input_str.split(',')]
    }
    
    return signature

def parse_recording(
        input_str: str,
        net_info: Dict[str, Tuple[int, ...]],
    ) -> Dict[str, RecordingUnit]:
    '''
    Converts a input string indicating the units associated to each layer
    to a dictionary mapping layer number to units indices.

    The format to specify the structure requires separating specification 
    for different layers with comma:
        layer1=units1, layer2=units2, ..., layerN=unitsN

    Where units specification can be:
        - all neurons; requires no specification:
            []
        - individual unit specification; requires neurons index to be separated by a space:
            [(A1 A2 A3) (B1 B2 B3) ...] <-- each neuron is identified by a tuple of numbers
            [A B C D ...]               <-- each neuron is identified by a single number, requiring no parenthesis
        - units from file; requires to specify a path to a .txt file containing one neuron per line
                           (each neuron is expected to be a set of integers separated by space):
            [neurons.txt]
        - A set of neurons in a given range with format FROM:TO:STEP; 
          it requires as many range specifications as neuron dimension:
            [A1_from:A1_to:A1_step A2_from:A2_to:A2_step ...]
        - random set of N neurons:
            Nr[]
        - random set of N neuron in a given range specified with FROM:TO;
          it requires as many range specifications as neuron dimension:
            Nr[A1_from:A1_to: A2_from:A2_to ...]
            
    :param input_str: The input string to parse.
    :type input_str: str
    :param net_info: Dictionary mapping layer names to its target unit.
    :type net_info: Dict[str, Tuple[int, ...]]
    '''
    
    # Split targets across layers
    # The dictionary `targets` maps the layer-ID to its string specification
    try:
        targets = {int(k.strip()) : v for k, v in [target.split('=') for target in input_str.split(',')]}
    except Exception as e:
            raise SyntaxError(f'Invalid specification in {input_str}: {e}.')

    # Output dictionary
    target_dict : Dict[str, RecordingUnit] = dict()

    # Extract layer names from the net information 
    # for mapping layer-IDs to their name
    layer_names = list(net_info.keys())

    # Process each layer iteratively
    for layer_idx, units in targets.items():
        
        units      = units.strip()            # remove spaces
        layer_name = layer_names[layer_idx]   # extract layer name
        shape      = net_info[layer_name][1:] # exclude batch size
        
        # Non-random specification immediately start with square bracket
        is_random = not units.startswith('[') 
        
        # A) Use all neurons
        if   not is_random and units=='[]': 
            neurons = None

        # B) Parsing from .TXT file
        elif not is_random and units.endswith('.txt]'):   
            neurons = neurons_from_file(file_path=units.strip('[]'))

        # C) Neurons in range
        elif not is_random and ':' in units:

            try:

                # Compute range bounds as a list of lists, 
                # - the first level is specific for neuron shape
                # - the second value contains the FROM, TO, STEP information
                # two elements (i.e. the bounds of the interval)

                bounds = [
                    [int(v) if v else None for v in tmp.split(':')] # None is to handle no full-specification
                    for tmp in units.strip('[]').split(' ')
                ]

                # Handle all possible bound combinations in the shape
                neurons = tuple(
                    np.array(
                        list(product(*list(starmap(range, bounds))))
                    ).T
                )

            except Exception as e:
                raise SyntaxError(f'Invalid range format specification in {units}: {e}.')
            
        # D) Single unit
        elif not is_random:

            try:

                # In the case of one-dimensional shape with no parenthesis 
                # we internally add to handle as tuple

                # NOTE: This implementation is only to allow a more plain
                #       specification to user in the case of a one dimensional layer

                if '(' not in units:
                    #e.g. [A B C] -> [(A) (B) (C)]
                    ranges = units.replace('[', '[(').replace(']', ')]').replace(' ', ') (')
                else:
                    ranges = units

                # Convert neuron to a tuple of arrays with:
                # - as many tuples as coordinates in the neuron shape
                # - array length as the number of neurons, each array codes for a particular
                #   coordinate of a neuron
                    
                neurons = tuple(np.array([
                        ([int(v) for v in code.strip('()').split(' ')])
                        for code in ranges.strip('[]').split(') (')
                    ]).T
                )

            except Exception as e:
                raise SyntaxError(f'Invalid units specification in {units}: {e}.')
            
        # E) Random units from interval
        elif is_random and ':' in units:

            try:
                
                # Splits the number of random units from the ranges of interest
                n_rand, ranges = units.split('r')
                n_rand = int(n_rand)
                
                bounds = [
                    [int(v) if v else None for v in tmp.split(':')] # None is to handle no full-specification
                    for tmp in ranges.strip('[]').split(' ')
                ]

                # Handle all possible bound combinations in the shape
                values = np.array(
                    list(product(*list(starmap(range, bounds))))
                )
                
                # NOTE: NumPy choice requires 1-dimensional arrays, so we
                #       sample the indices of the units we want, hence the
                #       need to store the values here
                neurons = tuple(
                    values[
                        np.random.choice(
                            a=len(values),
                            size=n_rand,
                            replace=False
                        )
                    ].T
                )
                
            except Exception as e:
                raise SyntaxError(f'Invalid random bounds specification in {units}: {e}.')
        
        # F) Unbound random units 
        elif is_random: 

            try:

                n_rand, _ = units.split('r')
                n_rand = int(n_rand)
                
                # Unravel indexes sampled in the overall interval
                neurons = np.unravel_index(
                    indices=np.random.choice(
                        a = np.prod(shape),
                        size=n_rand,
                        replace=False,
                    ),
                    shape=shape
                ) 
                    
            except Exception as e:
                raise SyntaxError(f'Invalid random bounds specification in {units}: {e}.')
            
        else:
            raise SyntaxError(f'Unrecognized specification in {units}. Please adhere to standard specification. ')
        
        # Check if neurons are out of bound
        if neurons and len([u for u in np.stack(neurons).T if tuple(u) > shape]): # type: ignore
            raise ValueError(f'Units out of bound for layer {layer_name} with shape {shape} in {units} specification. ')
        
        # Check if neurons have wrong dimension
        if neurons and len([u for u in np.stack(neurons).T if len(tuple(u)) != len(shape)]): # type: ignore
            raise ValueError(f'Units with different shape for layer {layer_name} with shape {shape} in {units} specification. ')

        # Add neurons to output dictionary
        target_dict[layer_name] = neurons

    return target_dict


def parse_scoring(
        input_str: str, 
        net_info: Dict[str, Tuple[int, ...]],
        rec_info: Dict[str, RecordingUnit]
    ) -> Dict[str, ScoringUnit]: 
    '''
    Converts an input string indicating the scoring units associated to each layer
    to a dictionary mapping layer name to a one-dimensional array of activations indexes
    referred to the corresponding recording targets.

    It also provides an optional second dictionary relative to the indexes of 
    recorded but non scored units.
    
    :param input_str: The input string to parse.
    :type input_str: str
    :param net_info: Dictionary mapping layer names to its target unit.
    :type net_info: Dict[str, Tuple[int, ...]]
    :param rec_info: Dictionary mapping layer names to recorded units in tuple form.
    :type rec_info: Dict[str, TargetUnit]
    '''

    def neuron_to_activation(layer_name: str, unit_: Tuple[int, ...]) -> int:

        neuron_mapping = lookup_table[layer_name]

        if neuron_mapping:
            unit_str = "_".join([str(u) for u in unit_])
            activation_idx =  neuron_mapping[unit_str]
        else:
            shape = net_info[layer_name][1:]
            activation_idx =  int(np.ravel_multi_index(
                #multi_index=tuple([np.array([el]) for el in unit_]), 
                multi_index=unit_, 
                dims=shape
            ))

        return activation_idx

    # TODO The random scoring with range is still not implemented
    if 'r' in input_str and ':' in input_str:
        raise NotImplementedError('Random scoring in range not supported yet.')
    
    # Unbounded random
    elif 'r' in input_str:

        # Extract targets mapping layers number to units specification
        try:
            targets = {int(k.strip()) : v for k, v in [target.split('=') for target in input_str.split(',')]}
        
        except Exception as e:
                raise SyntaxError(f'Invalid format in {input_str}: {e}')

        # Output
        scoring : Dict[str, ScoringUnit] = dict()
        layer_names = list(net_info.keys())

        for layer_idx, units in targets.items():

            layer_name = layer_names[layer_idx]

            # Extract number of random units
            rnd_units = int(units.split('r')[0])

            # Extract the number of recorder units
            rec_layer = rec_info[layer_name]
            rec_units = rec_layer[0].size if rec_layer else rnd_units  # `rec_layers` = None if recording from all units

            # Check on the number of units to score with respect to recorded ones
            if rnd_units > rec_units:
                raise ValueError(f'Trying to score {rnd_units} random units, but {rec_units} were recorded. ')
            
            # Sample activation index of scoring units
            scoring[layer_name] = list(
                np.random.choice(
                    a=np.arange(0,rec_units), 
                    size=rnd_units, 
                    replace=False
                )
            )
           
    else: 
    
        # In the deterministic case we use the same parsing as for the recording one
        score_target = parse_recording(
            input_str=input_str, 
            net_info=net_info
        )

        # Create a lookup table mapping neuron to its activation index
        # - the first level is the layer name
        # - the second level maps a stringfy version of the neuron coordinate (A1 A2 A3) --> A1_A2_A3 
        #   to its activation index 
        lookup_table: Dict[str, Dict[str, int] | None] = dict()

        for layer, units in rec_info.items():

            WARN_TRESHOLD = 10000

            # If units were specified we create a mapping
            if units:

                # Warning for memory allocation
                if len(units) > WARN_TRESHOLD:
                    warnings.warn(f"Trying to allocate a lookup table with more than {WARN_TRESHOLD} elements", UserWarning)
                
                # Maps neurons to their activation index
                lookup_table[layer] = {
                    '_'.join([str(n) for n in neurons]): i 
                    for i, neurons in enumerate(np.array(units).T)
                }

            # Otherwise all where specified, we don't allocate a lookup table since the mapping is trivial
            else:
                lookup_table[layer] = None
            
        # For each layer we store the activation 
        try:
            scoring = {
                layer: [
                    neuron_to_activation(unit_=tpl, layer_name=layer)
                    for tpl in np.array(units).T
                ] if units else []
                for layer, units in score_target.items()
            }
            
        except KeyError as e:
            raise ValueError('Trying to score non recorded neuron')

    return scoring
        
    
# TODO not here
def get_neurons_target(
        layers:  List[int],
        neurons: List[int],
        n_samples: int = 1,
        rec_both: bool = True,
        random_rec: int = 1000
        
) -> Tuple[str, str, str]:
        #flatten takes a list of lists and flattens it in one flat list
        def flatten(lst): return [item for sublist in lst for item in sublist]
        
        #return the correctly written parsing strings.[:-1] is to exclude the last # from the string
        targets = ''.join(flatten([[f'{l}={n}r[]#']*n_samples for l in layers  for n in neurons]))[:-1]
        linear_l = [layer for layer in layers if layer > 15]
        conv_layers = [layer for layer in layers if layer <= 15]
        rec_layers_lin = ''.join(flatten([[f'{l}=[]#']*(n_samples * len(neurons)) for l in linear_l ]))[:-1] if linear_l else ''

        rec_layers_conv = ''.join(flatten([[f'{l}={random_rec}r[]#']*(n_samples * len(neurons)) for l in conv_layers ]))[:-1] if conv_layers else ''

        rec_layers = '#'.join([rec_layers_lin,rec_layers_conv]) if rec_layers_lin and rec_layers_conv else rec_layers_lin or rec_layers_conv
        rand_seeds = '#'.join([str(np.random.randint(0, 100000)) for _ in range(n_samples * len(neurons) * len(layers))])
                
        if rec_both:
            rec_layers = '#'.join([','.join(list(set(rec_layers.split('#'))))]*(n_samples * len(neurons) * len(layers)))

        return targets, rec_layers, rand_seeds