import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

# --- TYPE ALIASES ---

Mask = List[bool]   
''' 
Boolean mask associated to a set of stimuli, to indicating if they refer 
to synthetic of natural images (True for synthetic, False for natural).
'''     

Codes = NDArray | Tensor
'''
Set of codes representing the images in a latent space.
The first dimension of the tensor is the batch size.
'''

Stimuli = Tensor
'''
Set of visual stimuli.
The first dimension of the tensor is the batch size.
'''

SubjectState = Dict[str, NDArray]
'''
Set of subject responses to a visual stimuli.
The subject state can have multiple layers, whose name 
is mapped to its specific activations in the form of a batched array.
'''

StimuliScore = NDArray[np.float32] 
'''
Set of scores associated to each stimuli in
the form of a one-dimensional array.
'''

# --- MESSAGE ---

@dataclass
class Message:
    '''
    The dataclass is an auxiliary generic component that
    is shared among the entire data-flow.
    The aim of the class is to make different components communicate
    through the data-passing of common object they all can manipulate.
    '''
    
    mask    : NDArray[np.bool_]
    '''
    Boolean mask associated to a set of stimuli indicating if they are
    synthetic of natural images.
    
    NOTE: The mask has not `Mask` type because it's not a list but an array.
          This is made to facilitate filtering operations that are primarily
          applied to arrays.
    '''
    
    label   : List[str] | None = None
    '''
    List of labels associated to the set of stimuli.
    '''
    

# --- Logger ---

# TODO - evolve the Logger into an IOHandler to save/load any type of data.
class Logger:
    '''
    Class responsible for logging in the three channels info, warn and error.
    
    NOTE: The logging methods can be easily overridden to log with other strategies 
          and technologies.
    '''

    def info(self,  mess: str): logging.info(mess)

    def warn(self,  mess: str): logging.warn(mess)

    def error(self, mess: str): logging.error(mess)