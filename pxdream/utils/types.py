'''
This module contains the type definitions used in the
'''

from typing      import Callable, Dict, List, Tuple

import numpy    as np
from numpy.typing import NDArray
from torch        import Tensor

# --- DATA FLOW ---

Codes = NDArray[np.float32]
'''
Set of codes representing the visual stimuli in a latent space.
They are represented with a 2-dimensional array.
The the first dimension the batch size (the number of codes) 
and the second dimension the code length.
'''


Stimuli = Tensor
'''
Set of visual stimuli. They are represented as a 4-dimensional tensor,
the first dimension is the batch size, while the other ones refers to 
width, height and channels that may depend on the specific generator.
'''


Layer = str
'''
Identifier for a layer in a generic ANN architecture
'''


Activations = NDArray[np.float32] 
'''
Units activation for a specific layer.
The array is two dimensional, with the first dimension as the batch 
dimension and the second one the number of recorded units.
'''


States = Dict[Layer, Activations]
'''
Set of subject responses to a visual stimuli.
The subject state can refer to multiple layers, with each layer
referring to a different entry in the dictionary that maps to
its specific activations in the form of a batched array 
NOTE: All the states in different layers must have the same batch size.
'''


Fitness = NDArray[np.float32] 
'''
Fitness associated to each individual in a population,
that is typically computed as a function of the states.

It is represented the form of a one-dimensional array whose
length is the population size.
'''

# --- MASK ---

Mask = NDArray[np.bool_]   
''' 
Boolean mask associated to a set of stimuli.
It indicates weather they refer to to synthetic stimuli 
of natural stimuli (True for synthetic, False for natural).
'''    

MaskGenerator = Callable[[int], Mask]
'''
Function producing a boolean mask given the number of synthetic images in a stimuli set.
The number of True values in the generated mask must correspond to the number of input synthetic images.
'''

# --- SCORER MAP and REDUCE ---

UnitsMapping = Callable[[Activations], Activations]
'''
Function implementing an activation mapping in any state of a recorded layer.
'''

UnitsReduction = Callable[[States], Dict[Layer, Fitness]]
'''
Reducing function across units of the same layer.
It computes a fitness for each individual in a population for the specific layer,
'''

LayerReduction = Callable[[Dict[Layer, Fitness]], Fitness]
'''
Reducing function across fitness for multiple layers.
It aggregates layer-specific fitness into a single fitness for each individual
'''

# --- RECORDING and SCORING UNITS ---

RecordingUnits = List[Tuple[int, ...]]
'''
Units to record from for a specific layers.
Since non-linear layers have multiple dimensions, 
the target unit must specify each of its dimensions as a tuple

Empty lists indicate to record from all the units in the layer.
'''

ScoringUnits = List[int]
'''
Structure describing the target unit to score from for a specific layer.
The index refers to the recorded units. 

NOTE: Differently from `RecordingUnits`, the unit specification it's one-dimensional 
    as it refers to the recording index.

Empty lists indicate to score from all the recorded units in the layer.
'''

# --- RECEPTIVE FIELDS ---

RFBox = Tuple[Tuple[int, int]]
'''
Receptive Field bounding box. Each tuple contains the extremes of 
the bounding box (each subtuple = vertices (2) in that dimension)
e.g. bounding box in a 3 channel image ((0,3),(134,145),(198,209))
the box covers all 3 color channels, and is a rectangle between 134 and 145 (h)
and 198 and 209 (w).
'''






