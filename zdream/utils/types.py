from typing      import Callable, Dict, List, Tuple

import numpy    as np
from numpy.typing import NDArray
from torch        import Tensor

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

States = Dict[str, NDArray]
'''
Set of subject responses to a visual stimuli.
The subject state can refer to multiple layers, whose name 
is mapped to its specific activations in the form of a batched array.
'''

Scores = NDArray[np.float32] 
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

UnitsMapping = Callable[[NDArray], NDArray]
'''
Mapping transformation for activations units
'''

UnitsReduction = Callable[[States], Dict[str, Scores]]
'''
Reducing function across units of the same layer
'''

LayerReduction = Callable[[Dict[str, Scores]], Scores]
'''
Reducing function across units of the same layer
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


