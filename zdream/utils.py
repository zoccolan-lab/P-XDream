import json
from einops import rearrange
import numpy as np
import torch.nn as nn
from typing import Tuple, TypeVar, Callable, Dict, List, Any, Union
import re
from numpy.typing import NDArray
import torch
import glob
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from PIL import Image
import random


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


# --- TORCH ---

# Default device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    img = np.asarray(img) / 255.
    img = rearrange(img, 'h w c -> 1 c h w')
    
    return img

