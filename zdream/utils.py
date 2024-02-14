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
ObjectiveFunction = Callable[[SubjectState], NDArray[np.float32]]

# Type function utils
def default(var : T | None, val : D) -> T | D:
    return val if var is None else var

def lazydefault(var : T | None, expr : Callable[[], D]) -> T | D:
    return expr() if var is None else var

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

def multioption_prompt(opt_list: List[str], in_prompt: str) -> Union[str, List[str]]:
    '''
    Prompt the user to choose from a list of options.

    Parameters:
    - opt_list: List of options.
    - in_prompt: Prompt message to display.

    Returns:
    - Either a single option (str) or a list of options (List[str]).
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

def multichar_split(my_string: str, separator_chars: List[str] =['-', '.'])-> List[str]:
    """
    Split a string using multiple separator characters.

    Args:
        my_string (str): The input string to be split.
        separator_chars (List[str]): List of separator characters. Default is ['-','.'].

    Returns:
        List[str]: List containing the substrings resulting from the split.
    """
    # Build the regular expression pattern to match any of the separator characters
    pattern = '[' + re.escape(''.join(separator_chars)) + ']'
    return re.split(pattern, my_string) # Split the string using the pattern as separator