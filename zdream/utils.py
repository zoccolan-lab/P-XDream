import numpy as np
import torch.nn as nn
from typing import TypeVar, Callable, Dict, List, Union
import re
from numpy.typing import NDArray
from torch import Tensor
from PIL.Image import Image

# Type Generics
T = TypeVar('T')
D = TypeVar('D')

# Type Aliases
Stimulus = Tensor | Image # TODO Or Stimuli = Tensor | List[Image] ? 
SubjectState = NDArray | Dict[str, NDArray]
SubjectScore = NDArray[np.float32] | Dict[str, NDArray[np.float32]]
ObjectiveFunction = Callable[[SubjectState], SubjectScore]


# Type function utils
def is_multiple_state(state: SubjectState) -> bool:
    return isinstance(state, dict)

def default(var : T | None, val : D) -> T | D:
    return val if var is None else var

def lazydefault(var : T | None, expr : Callable[[], D]) -> T | D:
    return expr() if var is None else var

# Torch function utils
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

    for c in children: unpacked.append(c)
    
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
