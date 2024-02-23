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

T = TypeVar('T')
D = TypeVar('D')

def exists(var: Any | None) -> bool:
    return var is not None

# Type function utils
def default(var : T | None, val : D) -> T | D:
    return val if var is None else var

def lazydefault(var : T | None, expr : Callable[[], D]) -> T | D:
    return expr() if var is None else var


# Torch function utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unpack(model : nn.Module) -> nn.ModuleList:
    '''
    Utils function to extract the layer hierarchy from a torch
    Module. This function recursively inspects each module
    children and progressively build the hierarchy of layers
    that is then return to the user.
    
    :param model: Torch model whose hierarchy we want to unpack
    :type model: torch.nn.Module
    
    :returns: List of sub-modules (layers) that compose the model
        hierarchy
    :rtype: nn.ModuleList
    '''
    children = [unpack(children) for children in model.children()]
    unpacked = [model] if list(model.children()) == [] else []

    for c in children: unpacked.extend(c)
    
    return nn.ModuleList(unpacked)

# String function utils
def multioption_prompt(opt_list: List[str], in_prompt: str) -> str | List[str]:
    '''
    Prompt the user to choose from a list of options

    :param opt_list: List of options.
    :type opt_list: List[str]
    :param in_prompt: Prompt message to display.
    :type in_prompt: str
    :return: Either a single option or a list of options.
    :rtype: str | List[str]
    '''
    
    # Generate option list
    opt_prompt = '\n'.join([f'{i}: {opt}' for i, opt in enumerate(opt_list)])
    
    # Prompt user and evaluate input
    idx_answer = eval(input(f"{in_prompt}\n{opt_prompt}"))
    
    # Check if the answer is a list
    if isinstance(idx_answer, list):
        answer = [opt_list[idx] for idx in idx_answer]
    else:
        # If not a list, return the corresponding option
        answer = opt_list[idx_answer]

    return answer


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

def xor(a: bool, b: bool) -> bool:
    return (a and b) or (not a and not b)

def read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found at path: {path}')
    
    
def repeat_pattern(
    n : int,
    base_seq: List[Any] = [True, False], 
    rand: bool = True
) -> List[Any]:
    bool_l = []; c = 0
    while c<n:
        if rand:
            random.shuffle(base_seq)
        bool_l = bool_l+base_seq
        c += sum(base_seq)    
    return bool_l

def logicwise_function(
    f: Union[Callable[[NDArray], NDArray], List[Callable[[NDArray], NDArray]]], 
    np_arr: NDArray, 
    np_l: NDArray
): # TODO: Add return type    
    if isinstance(f, list):
        results_l = tuple(f_func(np_arr[np_l]) for f_func in f)
        results_not_l = tuple(f_func(np_arr[~np_l]) for f_func in f)
    else:
        results_l = f(np_arr[np_l])
        results_not_l = f(np_arr[~np_l])
    
    return results_l, results_not_l

def preprocess_image(image_fp: str, resize: Tuple[int, int] | None)  -> NDArray:
    
    img = Image.open(image_fp).convert("RGB")
    if resize:
        img = img.resize(resize)
    img =np.asarray(img) / 255.
    img = rearrange(img, 'h w c -> 1 c h w')
    
    return img

