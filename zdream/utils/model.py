from dataclasses import dataclass
import tkinter as tk
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from PIL import Image, ImageTk

from typing import Tuple

import torch.nn as nn

# --- TYPE ALIASES ---

Mask = List[bool]   
''' 
Boolean mask associated to a set of stimuli, to indicating if they refer 
to synthetic of natural images (True for synthetic, False for natural).
'''     

Codes = NDArray[np.float64] #| Tensor
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

RFBox = Tuple[int, ...]
'''
Receptive Field bounding box, usually expected in
the form (x0, x1, y0, y1) but generalizable to
arbitrary number of dimensions.
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
    

class InputLayer(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x : Tensor) -> Tensor:
        return x

    def _get_name(self):
        return 'Input'

# --- LOGGER ---

class DisplayScreen:
	''' Screen to display and update images'''
    
	def __init__(self, title: str = "Image Display", display_size: Tuple[int, int] = (400, 400)):	
		'''
        Initialize a display window with name and size.

        :param title: Screen title, defaults to "Image Display"
		:type title: str, optional
		:param display_size: _description_, defaults to (400, 400)
		:type display_size: tuple, optional
		'''
		
		# Input parameters
		self._title  = title
		self._display_size = display_size
		
		# Screen master
		self._master = tk.Toplevel()
		self._master.title(title)
		
		# Create a container frame for the image label
		self._image_frame = tk.Frame(self._master)
		self._image_label = tk.Label(self._image_frame)
		
		self._image_frame.pack()
		self._image_label.pack()

	def update(self, image: Image.Image):
		'''
		Display the new image to the screen

		:param image: New image to be displayed.
		:type image: Image.Image
		'''
		
		# Resize the image to fit the desired display size
		resized_image = image.resize(self._display_size)
		
		# Convert the resized image to a Tkinter PhotoImage
		photo = ImageTk.PhotoImage(resized_image)
		
		# Configure the label with the resized image
		self._image_label.configure(image=photo)
		self._image_label.image = photo              # type: ignore
		
		# Update the frame
		self._master.update()

	def close(self):
		'''
		Method to be invoked to close the screen.
		NOTE: After this, the object becomes useless and a new one
			  needs to be instantiated.
		'''
		self._master.after(100, self._master.destroy) 