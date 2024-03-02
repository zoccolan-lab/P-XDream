import os
import re
import random
import json
from typing import Tuple, TypeVar, Callable, Dict, List, Any, Union, cast

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from einops import rearrange
import pandas as pd

from .model import RFBox

# --- TYPING ---

# Type generics
T = TypeVar('T')
D = TypeVar('D')

# Default for None with value
def default(var : T | None, val : D) -> T | D:
    return val if var is None else var

# Default for None with producer function
def lazydefault(var : T | None, expr : Callable[[], D]) -> T | D:
    return expr() if var is None else var

# --- NUMPY ---

def fit_bbox(
    data : NDArray | None,
    axes : Tuple[int, ...] = (-2, -1)
) -> RFBox:
    '''
    Fit a bounding box for non-zero entries of a
    numpy array along provided directions.
    
    :param grad: Array representing the data we want
        to draw bounding box over. Typical use case
        is the computed (input-)gradient of a subject
        given some hidden units activation states
    :type grad: Numpy array or None
    :param axes: Array dimension along which bounding
        boxes should be computed
    :type axes: Tuple of ints
    
    :returns: Computed bounding box in the format:
        (x1_min, x1_max, x2_min, x2_max, ...)
    :rtype: Tuple of ints
    '''
    if data is None: return (0, 0, 0, 0)
    
    # Get non-zero coordinates of gradient    
    coords = data.nonzero() 
    
    bbox = []
    # Loop over the spatial coordinates
    for axis in axes:
        bbox.extend((coords[axis].min(), coords[axis].max() + 1))
        
    return tuple(bbox)


def convert_to_numpy(data: Union[list, tuple, np.ndarray, torch.Tensor, pd.DataFrame]):
    """
    Converte un qualsiasi dato in un array NumPy.

    Parameters:
    - data: Il dato da convertire (può essere una lista, una tupla, un array NumPy, un tensore PyTorch o un DataFrame pandas).

    Returns:
    - numpy_array: L'array NumPy risultante.
    """
    try:
        if isinstance(data, pd.DataFrame):
            numpy_array = data.to_numpy()
        elif isinstance(data, torch.Tensor):
            numpy_array = data.numpy()
        else:
            numpy_array = np.array(data)
        return numpy_array
    except Exception as e:
        print(f"Errore durante la conversione in NumPy array: {e}")
        return None

# --- TORCH ---

# Default device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_nans(arr):
    '''
    Count the number of NaN values in a NumPy array or PyTorch tensor.

    :param arr: The input array or tensor.
    :type arr: numpy.ndarray or torch.Tensor
    :return: The number of NaN values in the array or tensor.
    :rtype: int
    '''
    if   isinstance(arr, np.ndarray):   return np.sum(np.isnan(arr))
    elif isinstance(arr, torch.Tensor): return torch.sum(torch.isnan(arr)).item()
    else: raise ValueError("Input must be a NumPy array or a PyTorch tensor.")

def unpack(model : nn.Module) -> nn.ModuleList:
    '''
    Utils function to extract the layer hierarchy from a torch Module.
    This function recursively inspects each module children and progressively 
    build the hierarchy of layers that is then return to the user.
    
    :param model: Torch model whose hierarchy we want to unpack.
    :type model: torch.nn.Module
    
    :returns: List of sub-modules (layers) that compose the model hierarchy.
    :rtype: nn.ModuleList
    '''
    
    children = [unpack(children) for children in model.children()]
    unpacked = [model] if list(model.children()) == [] else []

    for c in children: unpacked.extend(c)
    
    return nn.ModuleList(unpacked)

# NOTE: Code taken from github issue:
# https://github.com/pytorch/vision/issues/6699
def _replace_inplace(module : nn.Module) -> None:
    reassign = {}
    
    for name, mod in module.named_children(): 
        _replace_inplace(mod) 
        # Checking for explicit type instead of instance 
        # as we only want to replace modules of the exact type 
        # not inherited classes 
        if type(mod) is nn.ReLU or type(mod) is nn.ReLU6: 
            reassign[name] = nn.ReLU(inplace=False) 

    for key, value in reassign.items(): 
        module._modules[key] = value 

# --- STRING ---

def multichar_split(my_string: str, separator_chars: List[str] = ['-', '.'])-> List[str]:
    '''
    Split a string using multiple separator characters.

    :param my_string: The input string to be split.
    :type my_string: str
    :param separator_chars: List of separator characters, defaults to ['-', '.']
    :type separator_chars: List[str], optional
    :return: List containing the substrings resulting from the split.
    :rtype: List[str]
    '''
    
    # Build the regular expression pattern to match any of the separator characters
    pattern = '[' + re.escape(''.join(separator_chars)) + ']'
    
    # Split the string using the pattern as separator
    split = re.split(pattern, my_string) 
    
    return split
    
    
def repeat_pattern(
    n : int,
    base_seq: List[Any] = [True, False], 
    shuffle: bool = True
) -> List[Any]:
    '''
    Generate a list by repeating a pattern with shuffling option.

    :param n: The number of times to repeat the pattern.
    :type n: int
    :param base_seq: The base sequence to repeat, defaults to [True, False].
    :type base_seq: List[Any], optional
    :param rand: Whether to shuffle the base sequence before repeating, defaults to True.
    :type rand: bool, optional
    :return: A list containing the repeated pattern.
    :rtype: List[Any]
    '''
    
    bool_l = []
    
    for _ in range(n):
        if shuffle:
            random.shuffle(base_seq)
        bool_l.extend(base_seq)
        
    return bool_l

# --- I/O ---

def read_json(path: str) -> Dict[str, Any]:
    '''
    Read JSON data from a file.

    :param path: The path to the JSON file.
    :type path: str
    :raises FileNotFoundError: If the specified file is not found.
    :return: The JSON data read from the file.
    :rtype: Dict[str, Any]
    '''
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found at path: {path}')
    

def save_json(data: Dict[str, Any], path: str):
    '''
    Save JSON data to a file.

    :param data: The JSON data to be saved.
    :type data: Dict[str, Any]
    :param path: The path to save the JSON file.
    :type path: str
    '''
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    
def to_gif(image_list: List[Image.Image], out_fp: str, duration: int = 100):
    '''
    Save a list of input images as a .gif file.

    :param image_list: List of images to be saved as .gif file.
    :type image_list: List[Image.Image]
    :param out_fp: File path where to save the image.
    :type out_fp: str
    :param duration: Duration of image frame in milliseconds, defaults to 100.
    :type duration: int, optional
    '''

    image_list[0].save(
        out_fp, 
        save_all=True,
        optimize=False, 
        append_images=image_list[1:], 
        loop=0, duration=duration
    )

def rmdir(directory):
    """
    Recursively removes the contents of a directory and the directory itself.
    """

    # Iterate over the contents of the directory
    for item in os.listdir(directory):

        # Construct the full path of the item
        item_path = os.path.join(directory, item)
        
        # Check if the item is a file
        if os.path.isfile(item_path):
            os.remove(item_path)
        # If the item is a directory, recursively remove its contents
        elif os.path.isdir(item_path):
            rmdir(item_path)
    
    # After removing all contents, remove the directory itself
    os.rmdir(directory)

def numbers_from_file(file_path: str) -> List[int]:
    with open(file_path, 'r') as file:
        numbers = [int(line.strip()) for line in file]
    return numbers

# -- PARSING --

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



# --- IMAGES ---

def preprocess_image(image_fp: str, resize: Tuple[int, int] | None)  -> NDArray:
    '''
    Preprocess an input image by resizing and batching.

    :param image_fp: The file path to the input image.
    :type image_fp: str
    :param resize: Optional parameter to resize the image to the specified dimensions, defaults to None.
    :type resize: Tuple[int, int] | None
    :return: The preprocessed image as a NumPy array.
    :rtype: NDArray
    '''
    
    # Load image and convert to three channels
    img = Image.open(image_fp).convert("RGB")

    # Optional resizing
    if resize:
        img = img.resize(resize)
    
    # Array shape conversion
    img_arr = np.asarray(img) / 255.
    img_arr = rearrange(img_arr, 'h w c -> 1 c h w')
    
    return img_arr

def concatenate_images(img_list: List[Tensor], nrow: int = 2):
    
    grid_images = make_grid(img_list, nrow=nrow)
    grid_images = to_pil_image(grid_images)
    grid_images = cast(Image.Image, grid_images)
    
    return grid_images


def SEMf(data: Union[List[float], Tuple[float], np.ndarray], axis: int = 0):
    """
    Calcola lo standard error of the mean (SEM) per una sequenza di numeri.

    Parameters:
    - data: Sequenza di numeri (lista, tupla, array NumPy, ecc.).
    - axis: Asse lungo il quale calcolare la deviazione standard (valido solo per array NumPy).

    Returns:
    - sem: Standard error of the mean (SEM).
    """
    try:
        # Converte la sequenza in un array NumPy utilizzando la funzione convert_to_numpy
        data_array = convert_to_numpy(data)
        
        # Calcola la deviazione standard e il numero di campioni specificando l'asse se necessario
        std_dev = np.nanstd(data_array, axis=axis) if data_array.ndim > 1 else np.nanstd(data_array)
        sample_size = data_array.shape[axis]
        
        # Calcola lo standard error of the mean (SEM)
        sem = std_dev / np.sqrt(sample_size)
        
        return sem
    except Exception as e:
        print(f"Errore durante il calcolo dello standard error of the mean (SEM): {e}")
        return None

def stringfy_time(sec: int | float) -> str:
    ''' Converts number of seconds into a hour-minute-second string representation. '''

    sec = int(sec)

    hours = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60

    time_str = ""
    if hours > 0:
        time_str += f"{hours} hour{'s' if hours > 1 else ''}, "
    if minutes > 0:
        time_str += f"{minutes} minute{'s' if minutes > 1 else ''}, "
    time_str += f"{sec} second{'s' if sec > 1 else ''}"
    
    return time_str


def overwrite_dict(a: Dict, b: Dict) -> Dict:
    ''' 
    Overwrite keys of a nested dictionary A with those of a 
    second flat dictionary B is their values are not none.
    '''
    for key in a:
        if isinstance(a[key], Dict):
            overwrite_dict(a[key], b)
        elif key in b:
            a[key] = b[key]
    return a

def flatten_dict(d: Dict)-> Dict[Any, Any]:
    """
    Flatten a nested dictionary recursively to multiple levels.
    """
    flattened_dict = {}
    for k, v in d.items():
        if isinstance(v, Dict):
            nested_flattened = flatten_dict(v)
            for nk, nv in nested_flattened.items():
                flattened_dict[nk] = nv
        else:
            flattened_dict[k] = v
    return flattened_dict