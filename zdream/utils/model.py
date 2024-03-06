import random
import tkinter as tk
from dataclasses import dataclass
from typing      import Callable, Dict, List, Tuple
from functools   import partial

import numpy    as np
import torch.nn as nn
from numpy.typing import NDArray
from torch        import Tensor
from PIL          import Image, ImageTk


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

ScoringFunction   = Callable[[SubjectState], Dict[str, StimuliScore]]
'''
Function evaluating the StimuliScore for each layer-specific activations
in a SubjectState.
'''

AggregateFunction = Callable[[Dict[str, StimuliScore]], StimuliScore]
'''
Function aggregating the StimuliScore for different layer into
a single StimuliScore.
'''

MaskGenerator = Callable[[int], Mask]
'''
Function producing a boolean mask for an input number of synthetic images in a stimuli set.
The number of True values in the mask must correspond to the number of input synthetic images.
'''

TargetUnit = None | NDArray | Tuple[NDArray, ...]
'''
Structure for describing the target unit to record or score from.
'''

# --- SCORING and AGGREGATE FUNCTION TEMPLATES ---

scoring_functions: Dict[str, ScoringFunction] = {

}

aggregating_functions: Dict[str, AggregateFunction] = {
	'mean'  : lambda x: np.mean  (np.stack(list(x.values())), axis=0),
	'sum'   : lambda x: np.sum   (np.stack(list(x.values())), axis=0),
	'median': lambda x: np.median(np.stack(list(x.values())), axis=0),
}

# --- MASK GENERATOR

def mask_generator_from_template(
        template: List[bool] = [True], 
        shuffle: bool = False
    ) -> MaskGenerator:
	'''
	Function to produce a mask generator from a given template 
	with shuffling option. The template is expected to contain a 
	single True and an arbitrary number of False.
	
	:param template: Boolean template with one single True value,
					defaults to [True].
	:type template: List[bool]
	:param shuffle: If to shuffle the template, defaults to False.
	:type shuffle: bool
	:return: Function for generating mask for template for an arbitrary number of 
			synthetic images in a a set of stimuli.
	:rtype: MaskGenerator
	'''
      
	def repeat_pattern(
		n : int,
		template: List[bool], 
		shuffle: bool
	) -> List[bool]:
		'''
		Generate a list by repeating an input pattern with shuffling option.
		'''
		
		bool_l = []
		
		for _ in range(n):
			if shuffle:
				random.shuffle(template)
			bool_l.extend(template)
			
		return bool_l
	
	n_true = template.count(True)
	if n_true != 1:
		raise ValueError(f'Expected template to contain 1 True value, but {n_true} were found.')
	
	return partial(repeat_pattern, template=template, shuffle=shuffle)


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
    
    NOTE: The mask has not `Mask` as it's not a list but an array.
          This is made to facilitate filtering operations that are primarily
          applied to arrays.
    '''
    
    label   : List[int]
    '''
    List of labels associated to the set of stimuli.
    
    NOTE: Labels are only associated to natural images so they are
          they only refers to 'False' entries in the mask.
    '''

# --- NETWORKS --- 

class InputLayer(nn.Module):
    ''' Class representing a trivial input layer for an ANN '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x : Tensor) -> Tensor:
        return x

    def _get_name(self) -> str:
        return 'Input'

# --- SCREEN ---

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
		
		# Screen controller
		self._controller = tk.Toplevel()
		self._controller.title(self._title)
		
		# Create a container frame for the image label
		self._image_frame = tk.Frame(self._controller)
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
		self._controller.update()

	def close(self):
		'''
		Method to be invoked to close the screen.
            
		NOTE: After this, the object becomes useless as the garbage
              collector takes controler with no more reference.
              The object becomes useless and a new one needs to be instantiated.
		'''
		self._controller.after(100, self._controller.destroy) 
