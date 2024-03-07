from itertools import product, starmap
import random
from typing import Dict, List, Tuple

from .model import TargetUnit
from .io_ import neurons_from_file

import numpy as np
from numpy.typing import NDArray


def parse_boolean_string(boolean_str: str) -> List[bool]:
    ''' Converts a boolean string of T and F in a boolean list'''

    def char_to_bool(ch: str) -> bool:
        match ch:
            case 'T': return True
            case 'F': return False
            case _: raise ValueError('Boolean string must contain only T and F symbols.')

    return[char_to_bool(ch=ch) for ch in boolean_str]


def parse_layer_target_units(
        input_str: str,
        net_info: Dict[str, Tuple[int, ...]],
    ) -> Dict[str, TargetUnit]:
    '''
    Converts a input string indicating the units associated to each layer
    to a dictionary mapping layer number to units indices.

    The format to specify the structure is requires separating dictionaries with comma:
        layer1=units1, layer2=units2, ..., layerN=unitsN

    Where units specification can be:
        - all neurons; requires no specification:
            []
        - individual unit specification; requires neurons index to be separated by a space:
            [(A1 A2 A3) (B1 B2 B3) ...] <-- each neuron is identified by a tuple of numbers
            [A B C D ...]               <-- each neuron is identified by a single number
        - units from file; requires to specify a path to a .txt file containing one neuron per line
            (where each neuron is expected to be a tuple of ints separated by space or a single number):
            [file.txt]
        - A set of neurons in a given range:
            [A:B:step C:D:step]
        - random set of N neurons:
            Nr[]
        - random set of N neuron, in a given range:
            Nr[A:B C:D E]
    '''
    
    # Split targets separated by a comma
    try:
        targets = {int(k.strip()) : v for k, v in [target.split('=') for target in input_str.split(',')]}
    except Exception as e:
            raise SyntaxError(f'Invalid format in {input_str}. Detected: {e}.\nPlease follow parsing in documentation')

    # Output dictionary
    target_dict : Dict[str, TargetUnit] = dict()

    # targets = input_str.split(',')
    layer_names = list(net_info.keys())

    for layer_idx, units in targets.items():
        
        # Units parsing
        units = units.strip()
        lname = layer_names[layer_idx]
        shape = net_info[lname][1:]
        
        is_random = not units.startswith('[') 
        
        # Non random branch
        if   not is_random and units=='[]': neurons = None
        elif not is_random and units.endswith('.txt]'):
            # TODO: Check this function for new syntax
            neurons = neurons_from_file(file_path=units.strip('[]'))
        elif not is_random and ':' in units:
            try:

                bounds = [
                    [int(v) if v else None for v in tmp.split(':')]
                    for tmp in units.strip('[]').split(' ')
                ]

                neurons = tuple(
                    np.array(
                        list(product(*list(starmap(range, bounds))))
                    ).T
                )

            except Exception as e:
                raise SyntaxError(f'Error while parsing range format. Provided command was {units}. Exception was: {e}')
        elif not is_random:
            try:

                if '(' not in units:
                    units_ = units.replace('[', '[(').replace(']', ')]').replace(' ', ') (')
                else:
                    units_ = units

                neurons = tuple(np.array([
                        ([int(v) for v in code.strip('()').split(' ')])
                        for code in units_.strip('[]').split(') (')
                    ]).T
                )

            except Exception as e:
                raise SyntaxError(f'Error while parsing individual neuron format. Provided command was {units}. Exception was: {e}')
        elif is_random and ':' in units:
            try:
                size, codes = units.split('r')
                
                neurons = tuple(
                    np.random.randint(*[
                            [int(tmp.split(':')[axis]) for tmp in codes.strip('[]').split(' ')]
                            for axis in (0, 1)
                        ],
                        size=(int(size), len(shape)),
                    ).T
                )
                
            except Exception as e:
                raise SyntaxError(f'Error while parsing random range format. Provided command was {units}. Exception was: {e}')
        elif is_random:
            try:
                size, _ = units.split('r')
                
                neurons = tuple(
                        np.random.randint(
                            low=0,
                            high=shape,
                            size=(int(size), len(shape)),
                        ).T
                    )
            except Exception as e:
                raise SyntaxError(f'Error while parsing full random format. Provided command was {units}. Exception was: {e}')
            
        else:
            error_msg = '''
            Invalid input string;  valid formats are:
                layer1=[units1], layer2=[units2], ..., layerN: [unitsN]
            with units specified either as:
                - all neurons; requires no specification:
                    []
                - individual unit specification; requires neurons index to be separated by a space:
                    [(A1 A2 A3) (B1 B2 B3) ...] <-- each neuron is identified by a tuple of numbers
                    [A B C D ...]               <-- each neuron is identified by a single number
                - units from file; requires to specify a path to a .txt file containing one neuron per line
                    (where each neuron is expected to be a tuple of ints separated by space or a single number):
                    [file.txt]
                - A set of neurons in a given range:
                    [A:B:step C:D:step E]
                - random set of N neurons:
                    Nr[]
                - random set of N neuron, in a given range:
                    Nr[A:B C:D E]
            '''
            raise SyntaxError(error_msg)
        
        if neurons and len([u for u in np.stack(neurons).T if tuple(u) > shape]): # type: ignore
            raise ValueError(f'Units out of bound for layer {lname} with shape {shape}. Provided command was: {units}')
        
        if neurons and len([u for u in np.stack(neurons).T if len(tuple(u)) != len(shape)]): # type: ignore
            raise ValueError(f'Units with different shape for layer {lname} with shape {shape}. Provided command was: {units}')

        target_dict[lname] = neurons

    return target_dict

def parse_scoring_units(
        input_str: str, 
        net_info: Dict[str, Tuple[int, ...]],
        rec_neurons: Dict[str, TargetUnit]
    ) -> Dict[str, List[int]]:
    '''
    Converts a input string indicating the scoring units associated to each layer
    to a dictionary mapping layer name to a one-dimensional array of activations indexes
    referred to the corresponding recording targets.
    '''

    # In the case of both random units with range raise an error
    if 'r' in input_str and ':' in input_str:
        raise NotImplementedError('Random scoring in range not supported yet.')
    
    # In case it's random select random from recorded one
    if 'r' in input_str:

        # Extract layer and units
        try:
            targets = {int(k.strip()) : v for k, v in [target.split('=') for target in input_str.split(',')]}
        except Exception as e:
                raise SyntaxError(f'Invalid format in {input_str}. Detected: {e}.\nPlease follow parsing in documentation')

        # Output dictionary
        target_dict : Dict[str, List[int]] = dict()
        layer_names = list(net_info.keys())

        for layer_idx, units in targets.items():

            layer_name = layer_names[layer_idx]

            # At this point we know how random units to generate
            rnd_units = int(units.split('r')[0])

            rec_neuron = rec_neurons[layer_name]
            rec_units = rec_neuron[0].size if rec_neuron else rnd_units

            if rnd_units > rec_units:
                raise ValueError(f'Trying to score {rnd_units} random units, but {rec_units} were registers')

            target_dict[layer_name] = list(np.random.randint(0, high=rec_units, size=rnd_units, dtype=int))

        return target_dict
    
    # Otherwise we convert them as for the scoring
    score_target = parse_layer_target_units(input_str=input_str, net_info=net_info)

    # In the case of None explicitly map all its neurons
    rec_neurons = {
        k: v if v else tuple(np.indices(net_info[k][1:])[i].ravel() for i in range(len(net_info[k][1:])))
        for k, v in rec_neurons.items()
    }

    # We create a mapping
    rec_to_i = {
        k: {'_'.join([str(j) for j in tpl]): i for i, tpl in enumerate(np.array(v).T)}
        for k, v in rec_neurons.items()
    }

    scoring = {
        k: ['_'.join([str(j) for j in tpl])for tpl in np.array(v).T]
        for k, v in score_target.items()
    }

    # Out
    try:
        return {
            k: [rec_to_i[k][i] for i in v]
            for k, v in scoring.items()
        }
    except KeyError as e:
        raise ValueError('Trying to score non recorded neuron')