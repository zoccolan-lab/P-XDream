'''
This is a general purpose file containing utility functions that are used across the entire Zdream framework.
'''

from collections import defaultdict
from copy import deepcopy
import glob
import os
import platform
from os import path
from subprocess import PIPE, Popen
from typing import Tuple, TypeVar, Callable, Dict, List, Any, Union, cast
from torchvision import transforms
from scipy.spatial.distance import pdist


import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from einops import rearrange
from pandas import DataFrame
from numpy.typing import NDArray

from pxdream.utils.logger import Logger, SilentLogger



from .types import RFBox
import subprocess

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
    if data is None: return 0, 0, 0, 0  # type: ignore
    
    # Get non-zero coordinates of gradient    
    coords = data.nonzero() 
    
    bbox = []
    # Loop over the spatial coordinates
    for axis in axes:
        bbox.append((coords[axis].min(), coords[axis].max() + 1))
        
    return tuple(bbox)


def to_numpy(data: List | Tuple | Tensor | DataFrame) -> NDArray:
    '''
    Convert data from different formats into a numpy NDArray.
    
    :param data: Data structure to convert into a numpy array.
    :type data: List | Tuple | Tensor | DataFrame
    
    :return: Data converted into a numpy array.
    :rtype: NDArray
    '''

    try:
        if isinstance(data, DataFrame):
            numpy_array = data.to_numpy()
        elif isinstance(data, Tensor):
            numpy_array = data.numpy()
        elif isinstance(data, List) or isinstance(data, Tuple):
            numpy_array = np.array(data)
        else:
            raise TypeError(f'Invalid input type {type(data)} for array conversion. ')
        return numpy_array
    
    except Exception as e:
        raise RuntimeError(f"Error during numpy array conversion: {e}")

# --- TORCH ---

# Default device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InputLayer(nn.Module):
    ''' Class representing a trivial input layer for an ANN '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x : Tensor) -> Tensor:
        return x

    def _get_name(self) -> str:
        return 'Input'

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

# NOTE: Code taken from github issue: https://github.com/pytorch/vision/issues/6699
def replace_inplace(module : nn.Module) -> None:
    '''
    Recursively replaces instances of nn.ReLU and nn.ReLU6 modules within a given
    nn.Module with instances of nn.ReLU with inplace=False.

    :param module: The PyTorch module whose ReLU modules need to be replaced.
    :type module: nn.Module
    '''

    reassign = {}
    
    for name, mod in module.named_children(): 

        replace_inplace(mod) 

        # NOTE: Checking for explicit type instead of instance 
        #       as we only want to replace modules of the exact type 
        #       not inherited classes 
        if type(mod) is nn.ReLU or type(mod) is nn.ReLU6: 
            reassign[name] = nn.ReLU(inplace=False) 

    for key, value in reassign.items(): 
        module._modules[key] = value 

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

def concatenate_images(img_list: Tensor | List[Tensor], nrow: int = 2) -> Image.Image:
    ''' 
    Concatenate an input number of images as tensors into a single image
    with the specified number of rows.
    '''
    
    grid_images = make_grid(img_list, nrow=nrow)
    grid_images = to_pil_image(grid_images)
    grid_images = cast(Image.Image, grid_images)
    
    return grid_images

# --- STATISTICS

def SEM(
        data: List[float] | Tuple[float] | NDArray, 
        axis: int = 0
    ) -> NDArray:
    '''
    Compute standard error of the mean (SEM) for a sequence of numbers

    :param data: Data to which compute the statistics.
    :type data: List[float] | Tuple[float]
    :param axis: Axis to compute SEM (valid for NDArray data only), default to zero.
    :type axis: int
    :return: Computed statistics SEMs
    :rtype: NDArray
    '''

    # Convert data into NDArray
    if not isinstance(data, np.ndarray):
        data = to_numpy(data)
    
    # Compute standard deviation and sample size on the specified axis
    std_dev = np.nanstd(data, axis=axis) if data.ndim > 1 else np.nanstd(data)
    sample_size = data.shape[axis]
    
    # Compute SEM
    sem = std_dev / np.sqrt(sample_size)

    return sem


def harmonic_mean(a: float, b: float) -> float:
    '''
    Compute the harmonic mean of two given numbers.

    :param a: First number
    :type a: float
    :param b: Second number
    :type b: float
    :return: Harmonic mean of the two numbers.
    :rtype: float
    '''
    
    
    return  2 / (1 / a + 1 / b) 


def growth_scale(start=0, step=0.05, growth_factor=1.5):
    '''
    Generate an infinite scale of growth sequence using a generator.

    :param start: Starting value of the sequence, defaults to 0.
    :type start: float, optional
    :param step: Step size between terms, defaults to 0.05.
    :type step: float, optional
    :param growth_factor: Growth factor between terms, defaults to 1.5.
    :type growth_factor: float, optional
    :yield: Next value in the growth scale sequence.
    :rtype: float
    '''
    
    current_value = start
    yield current_value
    while True:
        current_value += step
        step *= growth_factor
        yield current_value


# --- TIME ---

def stringfy_time(sec: int | float) -> str:
    ''' Converts number of seconds into a hour-minute-second string representation. '''

    # Round seconds
    sec = int(sec)

    # Compute hours, minutes and seconds
    hours = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60

    # Handle possible formats
    time_str = ""
    if hours > 0:
        time_str += f"{hours} hour{'s' if hours > 1 else ''}, "
    if minutes > 0:
        time_str += f"{minutes} minute{'s' if minutes > 1 else ''}, "
    time_str += f"{sec} second{'s' if sec > 1 else ''}"
    
    return time_str

# --- DICTIONARIES ---

def overwrite_dict(a: Dict, b: Dict) -> Dict:
    ''' 
    Overwrite keys of a nested dictionary A with those of a 
    second flat dictionary B is their values are not none.
    '''

    def overwrite_dict_aux(a_: Dict, b_: Dict):
    
        for key in a_:
            if isinstance(a_[key], Dict):
                overwrite_dict_aux(a_[key], b_)
            elif key in b_:
                a_[key] = b_[key]
        return a_
    
    # Create new dictionary
    a_copy = deepcopy(a)

    overwrite_dict_aux(a_=a_copy, b_=b)
    
    return a_copy

def flatten_dict(d: Dict)-> Dict:
    '''
    Recursively flat an input nested dictionary.
    '''

    flattened_dict = {}
    for k, v in d.items():
        if isinstance(v, Dict):
            nested_flattened = flatten_dict(v)
            for nk, nv in nested_flattened.items():
                flattened_dict[nk] = nv
        else:
            flattened_dict[k] = v
    return flattened_dict

# --- EXECUTION ---

# NOTE: This requires `sudo apt install xsel`
def copy_on_clipboard(command: str):
    ''' Copies input string to clipboard '''

    match platform.system():

        case 'Linux':

            # Byte conversion
            cmd_ =  bytes(command, encoding='utf-8')
            
            # Copy
            p = Popen(['xsel', '-bi'], stdin=PIPE)
            p.communicate(input=cmd_)

        case 'Windows':

            subprocess.run(f'echo {command} | clip', shell=True)


def copy_exec(
        file: str,
        program: str = 'python',
        args: Dict[str, str] = dict(),
    ) -> str:
    ''' 
    Copies a program execution command line to clipboard given 
    program name, file name and the list of name-value arguments in dictionary form
    It returns the command string
    '''
    
    cmd = f'{program} {file} ' + " ".join(f'--{k} {v}' for k, v in args.items())

    copy_on_clipboard(cmd)

    return cmd

def minmax_norm(vector):
    min_val = np.min(vector)
    return (vector - min_val) / (np.max(vector) - min_val)

def defaultdict_list():
    return defaultdict(list)


def load_npy_npz(in_dir:str, fnames:list[tuple[str, str]], logger: Logger = SilentLogger()):
    '''
    Load a .npy/.npz file (e.g. a experiment state) from a folder where the file
    was dumped. It raises a warning for not present states.

    :param in_dir: Directory where states are dumped.
    :type in_dir: str
    :param fnames: List of file names to load, structured as
        tuples (fname, extension).
    :type fnames: list[tuple[str, str]]
    :param logger: Logger to log i/o information. If not specified
        a `SilentLogger` is used. 
    :type logger: Logger | None, optional
    '''
    logger.info(f'Loading experiment state from {in_dir}')

    loaded = dict()
    for name, ext in fnames:

        # File path
        fp = path.join(in_dir, f'{name}.{ext}')

        # Loading function depending on file extension
        match ext:
            case 'npy': load_fun = np.load
            case 'npz': load_fun = lambda x: dict(np.load(x))

        # Loading state
        if path.exists(fp):
            logger.info(f"> Loading {name} history from {fp}")
            loaded[name] = load_fun(fp)

        # Warning if the state is not present
        else:
            logger.warn(f"> Unable to fetch {fp}")
            loaded[name] = None
    return loaded

