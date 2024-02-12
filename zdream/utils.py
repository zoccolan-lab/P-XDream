import numpy as np
from typing import TypeVar, Callable, Dict
from numpy.typing import NDArray, ArrayLike

T = TypeVar('T')
D = TypeVar('D')

SubjectState = NDArray | Dict[str, NDArray]
ObjectiveFunction = Callable[[SubjectState], NDArray[np.float32]]

def default(var : T | None, val : D) -> T | D:
    return val if var is None else var

def lazydefault(var : T | None, expr : Callable[[], D]) -> T | D:
    return expr() if var is None else var