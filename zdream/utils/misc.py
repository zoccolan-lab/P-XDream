import re
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
from pandas import DataFrame
from numpy.typing import NDArray


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

def concatenate_images(img_list: List[Tensor], nrow: int = 2) -> Image.Image:
    ''' 
    Concatenate an input number of images as tensors into a single image
    with the specified number of rows.
    '''
    
    grid_images = make_grid(img_list, nrow=nrow)
    grid_images = to_pil_image(grid_images)
    grid_images = cast(Image.Image, grid_images)
    
    return grid_images

# --- STATISTICS

def SEMf(
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
    
    for key in a:
        if isinstance(a[key], Dict):
            overwrite_dict(a[key], b)
        elif key in b:
            a[key] = b[key]
    return a

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