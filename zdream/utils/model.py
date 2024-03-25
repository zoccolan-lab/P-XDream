import tkinter as tk
from typing      import Callable, Dict, List, Tuple
from functools   import partial

import numpy    as np
import torch.nn as nn
from numpy.typing import NDArray
from torch        import Tensor
from PIL          import Image, ImageTk


# --- TYPE ALIASES ---

Mask = NDArray[np.bool_]   
''' 
Boolean mask associated to a set of stimuli, indicating if they refer 
to synthetic of natural images (True for synthetic, False for natural).
'''     

Codes = NDArray[np.float32]
'''
Set of codes representing the images in a latent space.
The representation is a 2-dimensional array, with the
first dimension the batch size (the number of codes)
and the second dimension the code length.
'''

Stimuli = Tensor
'''
Set of visual stimuli.
The representation is a 4-dimensional tensor,
the first dimension is the batch size.
'''

SubjectState = Dict[str, NDArray]
'''
Set of subject responses to a visual stimuli.
The subject state can refer to multiple layers, whose name 
is mapped to its specific activations in the form of a batched array.
'''

StimuliScore = NDArray[np.float32] 
'''
Set of scores associated to each stimuli in
the form of a one-dimensional array of size
equal to the batch size.
'''

RFBox = Tuple[Tuple[int, int]]
'''
Receptive Field bounding box. Each tuple contains the extremes of 
the bounding box (each subtuple = vertices (2) in that dimension)
e.g. bounding box in a 3 channel image ((0,3),(134,145),(198,209))
the box covers all 3 color channels, and is a rectangle between 134 and 145 (h)
and 198 and 209 (w).
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
Function producing a boolean mask given the number of synthetic images in a stimuli set.
The number of True values in generated mask must correspond to the number of input synthetic images.
'''

RecordingUnit = None | Tuple[NDArray, ...]
'''
Structure for describing the target unit to record from.
'''

ScoringUnit = List[int]
'''
Structure describing the target unit to score from.
'''

# --- MASK GENERATOR

def mask_generator_from_template(
        template: List[bool] = [True], 
        shuffle: bool = False
    ) -> MaskGenerator:
	'''
	Function to produce a mask generator from a given template 
	with shuffling option. The template is expected to contain a 
	single True value and an arbitrary number of False.
	
	:param template: Boolean template with one single True value,
					 defaults to [True].
	:type template: List[bool]
	:param shuffle: If to shuffle the template, defaults to False.
	:type shuffle: bool
	:return: Function for generating mask for template for an arbitrary number of 
		     synthetic images in a set of stimuli.
	:rtype: MaskGenerator
	'''
    
	def repeat_pattern(n : int, template: List[bool], shuffle: bool	) -> Mask:
		''' Generate a list by concatenating an input pattern with shuffling option. '''
		
		bool_l = []
		
		for _ in range(n):
			if shuffle:
				np.random.shuffle(template)
			bool_l.extend(template)
			
		return np.array(bool_l, dtype=np.bool_)
	
	# Check the presence of one single True value
	n_true = template.count(True)
	if n_true != 1:
		err_msg = f'Expected template to contain 1 True value, but {n_true} were found.'
		raise ValueError(err_msg)
	
	return partial(repeat_pattern, template=template, shuffle=shuffle)


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

	@staticmethod
	def set_main_screen() -> tk.Tk:
		'''
		Create the main screen for displaying images.
		The main must be created in order to render images in TopLevel additional screens.

		NOTE: The object returned by the function should be saved in a variable
		      that mustn't become out of scope; in the case it may be removed by 
			  the garbage collector and invalidate screen rendering process.
		'''

		main_screen = tk.Tk()
		main_screen.withdraw()  # hide the main screen

		return main_screen
    
	def __init__(self, title: str = "Image Display", display_size: Tuple[int, int] = (400, 400)):	
		'''
        Initialize a display window with title and size.

        :param title: Screen title, defaults to "Image Display"
		:type title: str, optional
		:param display_size: _description_, defaults to (400, 400)
		:type display_size: tuple, optional
		'''
		
		# Input parameters
		self._title        = title
		self._display_size = display_size
		
		# Screen controller
		self._controller = tk.Toplevel()
		self._controller.title(self._title)
		
		# Create a container frame for the image label
		self._image_frame = tk.Frame(self._controller)
		self._image_label = tk.Label(self._image_frame)
		
		self._image_frame.pack()
		self._image_label.pack()

	def __str__ (self) -> str: return self._title
	def __repr__(self) -> str: return str(self)


	def update(self, image: Image.Image):
		'''
		Display a new image to the screen

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
		Method to close the screen.
            
		NOTE: After this, the object becomes useless as the garbage
              collector takes controller with no more reference.
              The object becomes useless and a new one needs to be instantiated.
		'''
		self._controller.after(100, self._controller.destroy) 
