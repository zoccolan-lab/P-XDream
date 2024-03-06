import random
from typing import Dict, List, Tuple

from .model import TargetUnit
from .io_ import numbers_from_file

import numpy as np


def parse_boolean_string(boolean_str: str) -> List[bool]:
    ''' Converts a boolean string of T and F in a boolean list'''

    def char_to_bool(ch: str) -> bool:
        match ch:
            case 'T': return True
            case 'F': return False
            case _: raise ValueError('Boolean string must contain only T and F symbols.')

    return[char_to_bool(ch=ch) for ch in boolean_str]


def parse_layer_target_units(input_str: str, net_info: dict[str, Tuple[int, ...]]) -> Dict[str, TargetUnit]:
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
            [A:B:step C:D:step E]
        - random set of N neurons:
            Nr[]
        - random set of N neuron, in a given range:
            Nr[A:B C:D E]
    '''
    
    # Split targets separated by a comma
    try:
        targets = {int(k.strip()) : v for tmp in input_str.split(',') for k, v in tmp.split('=')}
    except Exception as e:
            raise SyntaxError(f'Invalid format in {input_str}. Detected: {e}.\nPlease follow parsing in documentation')

    # Output dictionary
    target_dict : dict[str, TargetUnit] = dict()

    # targets = input_str.split(',')
    layer_names = list(net_info.keys())

    for layer_idx, units in targets.items():
        
        # Units parsing
        units = units.strip()
        lname = layer_names[layer_idx]
        shape = net_info[lname]
        
        is_random = not units.startswith('[') 
        
        # Non random branch
        if   not is_random and units=='[]': neurons = None
        elif not is_random and units.endswith('.txt'):
            # TODO: Check this function for new syntax
            neurons = numbers_from_file(file_path=units)
        elif not is_random and ':' in units:
            try:
                neurons = tuple(np.arange(*[int(v) if v else None for v in tmp.split(':')], dtype=int)
                    for tmp in units.split(' ')
                )
            except Exception as e:
                raise SyntaxError(f'Error while parsing range format. Provided command was {units}. Exception was: {e}')
        elif not is_random:
            try:
                neurons = tuple(np.array([[int(v) for v in code.strip('()').split(' ')]
                        for code in units.split(' ')
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
        
        if neurons and len([u for u in neurons if u > shape]):
            raise ValueError(f'Units out of bound for layer {lname} with shape {shape}. Provided command was: {units}')

        target_dict[lname] = neurons

    return target_dict