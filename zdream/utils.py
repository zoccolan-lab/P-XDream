import glob
import json
from os import path
import os
import random
import re
from typing import Tuple, TypeVar, Callable, Dict, List, Any, Union, cast


import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from einops import rearrange
import pandas as pd
from loguru import logger

from .model import Logger

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

# --- LOGGERS ---

class LoguruLogger(Logger):
    """ Logger overriding logger methods with `loguru` ones"""
    
    def info(self, mess: str): logger.info(mess)
    def warn(self, mess: str): logger.warning(mess)
    def err (self, mess: str): logger.error(mess)

# --- DATASETS ---

"""
NOTE: TO download
Dataset from here: [https://www.kaggle.com/datasets/arjunashok33/miniimagenet?resource=download]
Classes from here: [https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57#file-map_clsloc-txt]
"""
class MiniImageNet(ImageFolder):

    def __init__(
        self,
        root: str,
        transform: Callable[..., Tensor] = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.ToTensor()
        ]), 
        target_transform=None
    ):
        
        super().__init__(
            root=root, 
            transform=transform,
            target_transform=target_transform
        )
        
        # Load the .txt file containing ImageNet labels (all 1000 categories)
        lbls_txt = glob.glob(path.join(root, '*.txt'))[0]
        
        with open(lbls_txt, "r") as f:
            lines = f.readlines()
        
        # Create a label dictionary
        self.label_dict = {
            line.split()[0]: line.split()[2].replace('_', ' ')
            for line in lines
        }
    
    # TODO Maintain this method here?
    def class_to_lbl(self, lbls : Tensor): 
        # Takes in input the labels and outputs their categories
        return [self.label_dict[self.classes[lbl]] for lbl in lbls.tolist()]
    
    def __getitem__(self, index: int) -> Tensor:
        # TODO This is  made to make the dataset work
        
        return torch.tensor(np.random.rand(3, 256, 256), dtype=torch.float32)
        
        # Select random image
        random_subfolder = random.choice([f.path for f in os.scandir(self.root) if f.is_dir()])
        random_image = random.choice([f.path for f in os.scandir(random_subfolder) if f.is_file()])
        
        image = preprocess_image(image_fp=random_image, resize=(256, 256))
        image_t = torch.tensor(image[0], dtype=torch.float32)
        
        return image_t
        # return super().__getitem__(index)[0]
    
class RandomImageDataset(Dataset):
    '''
    Random image dataset to simulate natural images to be interleaved
    in the stimuli with synthetic ones.
    '''
    
    def __init__(self, n_img: int, img_size: Tuple[int, ...]):
        self.n_img = n_img
        self.image_size = img_size
    
    def __len__(self):
        return self.n_img
    
    def __getitem__(self, idx) -> Tensor:
        
        # Simulate finite dataset
        if idx < 0 or idx >= len(self): raise ValueError(f"Invalid image idx: {idx} not in [0, {len(self)})")

        rand_img = torch.tensor(np.random.rand(*self.image_size), dtype=torch.float32)
        
        return rand_img

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
    img_arr = np.asarray(img) / 255.
    img_arr = rearrange(img_arr, 'h w c -> 1 c h w')
    
    return img_arr

def concatenate_images(img_list: List[Tensor], nrow: int = 2):
    
    grid_images = make_grid(img_list, nrow=nrow)
    grid_images = to_pil_image(grid_images)
    grid_images = cast(Image.Image, grid_images)
    
    return grid_images

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
