import random
from typing import Dict, List, Tuple

from zdream.utils.io_ import numbers_from_file

import numpy as np


def parse_boolean_string(boolean_str: str) -> List[bool]:
    ''' Converts a boolean string of T and F in a boolean list'''

    def char_to_bool(ch: str) -> bool:
        match ch:
            case 'T': return True
            case 'F': return False
            case _: raise ValueError('Boolean string must contain only T and F symbols.')

    return[char_to_bool(ch=ch) for ch in boolean_str]


def parse_layer_target_units(input_str: str, input_dim: Tuple[int, ...]) -> Dict[int, Tuple[int, ...]]:
    '''
    Converts a input string indicating the units associated to each layer
    to a dictionary mapping layer number to units indices.

    The format to specify the structure is requires separating dictionaries with comma:
        layer1: [units1], layer2: [units2], ..., layerN: [unitsN]

    Where units specification can be:
        - individual unit specification; requires neurons index to be separated by a space:
            [A B ... C]
        - units from file; requires to specify a path to a .txt file containing one neuron number per line:
            [file.txt]
        - range specification; requires to specify start and end neurons in a range separated by an underscore:
            [A_B]
        - random set of neurons; requires the number of random units followed by an r:
            [Ar]
        - all neurons; requires no specification:
            []
    '''

    # Total input units across dimensions
    tot_in_units = np.prod(input_dim)

    def neuron_to_input_shape(unit: int) -> Tuple[int, ...]:
        '''
        The function converts flat input number to the input dimensionality
        '''

        # Check unit incompatible with input dimension
        if unit > tot_in_units: raise ValueError(f'Invalid unit: {unit}')

        # Convert to tuple version
        out_unit = []

        for dim in input_dim:
            out_unit.append(unit % dim)
            unit = unit // dim

        return tuple(out_unit)

    # Output dictionary
    target_dict = dict()

    # Split targets separated by a comma
    targets = input_str.split(',')

    for target in targets:

        # Layer units split
        try:
            layer, units = target.split(':')
        except ValueError as e:
            raise SyntaxError(f'Invalid format in {target}. Expected a single `:`.')

        # Layer and units parsing
        layer = int(layer.strip())        # Layer cast to int
        units = units.strip("[] ") # Units without square brackets

        # 1. Range
        if '_' in units:
            try:
                low, up = [int(v) for v in units.split('_')]
                neurons = list(range(low, up))
            except ValueError:
                raise SyntaxError(f'Invalid format in {units}. Expected a single `_`.')

        # 2. File
        elif units.endswith('.txt'):
            neurons = numbers_from_file(file_path=units)

            neurons_err = [u for u in neurons if u > tot_in_units]
            if len(neurons_err):
                raise ValueError(f'Trying to record {neurons_err} units, but there\'s a total of {tot_in_units}.')


        # 3. Random
        elif units.count('r') == 1:

            try:
                n_rand = int(units.strip('r'))
            except ValueError:
                raise SyntaxError(f'Invalid format in {units}. Expected the number of random units followed by an `t`.')

            if n_rand > tot_in_units:
                raise ValueError(f'Trying to generate {n_rand} random units from  a total of {tot_in_units}.')

            neurons = random.sample(range(tot_in_units), n_rand)

        # 4. All neurons
        elif units.replace(' ', '') == '':

            neurons = None

        # 5. Specific neurons
        elif all(ch.isdigit() for ch in units.replace(' ', '')):

            neurons = [int(unit) for unit in units.split()]# if unit.strip()]

            neurons_err = [u for u in neurons if u > tot_in_units]
            if len(neurons_err):
                raise ValueError(f'Trying to record {neurons_err} units, but there\'s a total of {tot_in_units}.')

        else:
            error_msg = '''
            Invalid input string;  valid formats are:
                layer1: [units1], layer2: [units2], ..., layerN: [unitsN]
            with units specified either as:
                - single units:    [A B ... C]
                - units from file: [neurons.txt]
                - range units:     [A_B]
                - random units:    [Ar]
                - all units:       []
            '''
            raise SyntaxError(error_msg)

        # Convert inputs to input dimension
        # if neurons:
        #     neurons = [neuron_to_input_shape(n) for n in neurons] TODO What to do for structured neurons

        target_dict[layer] = neurons

    return target_dict