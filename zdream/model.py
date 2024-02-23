# Type Aliases
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from numpy.typing import NDArray

from torch import Tensor


Mask = List[bool]
Stimuli = Tensor
Codes = NDArray | Tensor
StimuliScore = NDArray[np.float32]  # 1-dimensional array with the length of the batch assigning a score to each tested stimulus
SubjectState = Dict[str, NDArray]   # State of a subject mapping each layer to its batch of activation


@dataclass
class Message:
    mask    : NDArray[np.bool_]
    label   : List[str] | None = None


class Logger:

    def info(self,  mess: str): print(f"INFO: {mess}")

    def warn(self,  mess: str): print(f"WARN: {mess}")

    def error(self, mess: str): print(f"ERR:  {mess}")