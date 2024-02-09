from typing import TypeVar, Callable

T = TypeVar('T')
D = TypeVar('D')

def default(var : T | None, val : D) -> T | D:
    return val if var is None else var

def lazydefault(var : T | None, expr : Callable[[], D]) -> T | D:
    return expr() if var is None else var

# def lazydefault(var : T | None, expr : Callable[[], D], val : D) -> T | D:
#     return expr() if var is None else var     