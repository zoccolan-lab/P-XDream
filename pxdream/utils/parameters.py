from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Type

from pxdream.utils.io_ import save_json

Parameter      = str | int | float | bool
''' '''


ParameterType  = Type[Parameter]

@dataclass
class ArgParam:

    name    : str
    help    : str
    type    : ParameterType
    default : Any = None
    
    def __hash__(self)        -> int:  return hash(self.name)
    def __eq__  (self, other) -> bool: return self.name == other.name if isinstance(other, ArgParam) else False
    
    @staticmethod
    def argconf_to_json(argconf: ParamConfig) -> Dict[str, Parameter]:
        ''' Convert the argument configuration to JSON-like dictionary. '''
        
        return {arg.name : val for arg, val in argconf.items()}
    
    @staticmethod
    def save_argconf(argconf: ParamConfig, fp: str = 'config.json') -> None:
        ''' Save the argument configuration to the JSON file. '''
        
        save_json(data=ArgParam.argconf_to_json(argconf), path=fp)

ParamConfig  = Dict[ArgParam, Parameter]


class ArgParams(Enum):

    # Logger
    ExperimentName     = ArgParam(name="name",               type=str,   help="Experiment name")
    ExperimentVersion  = ArgParam(name="version",            type=int,   help="Experiment version")
    OutputDirectory    = ArgParam(name="out_dir",            type=str,   help="Path to directory to save outputs")

    # Globals
    NumIterations      = ArgParam(name="iter",               type=int,   help="Number of total iterations")
    DisplayPlots       = ArgParam(name="display_plots",      type=bool,  help="If to display plots")
    RandomSeed         = ArgParam(name="random_seed",        type=int,   help="Random state for the experiment")
    Render             = ArgParam(name="render",             type=bool,  help="If to render stimuli")
    
    # Additional
    ExperimentTitle    = ArgParam(name="title",              type=str,   help='')  # NOTE: Not a proper argument, depends on the experiment run
    DisplayScreens     = ArgParam(name="display_screens",    type=bool,  help='')  # NOTE: Not a proper argument, key to pass a shared one-instance screen to multiple experiments
    CloseScreen        = ArgParam(name="close_screen",       type=bool,  help='')  # NOTE: Not a proper argument, key to indicate if to close the display screen after the experiment
    
    # --- MAGIC METHODS ---
    
    def __str__ (self)  -> str: return self.value.name
    def __repr__(self)  -> str: return str(self)
    
    # --- UTILITIES ---
        
    @staticmethod
    def get_parser(
        args:     List[ArgParam] = [],
        multirun: bool      = False
    ) -> ArgumentParser:
        ''' Return the argument parser with the input parameters. '''
        
        parser = ArgumentParser()
        
        for arg in args:
            
            parser.add_argument(
                f"--{arg.name}", 
                type=arg.type if not multirun else str, 
                help=arg.help, 
                default=arg.default
            )
        
        return parser
    
    @classmethod
    def from_str(cls, name: str) -> ArgParam:
        ''' Return the argument from the string name. '''
        
        for arg in cls:
            if str(arg) == name: return arg.value
        
        raise ValueError(f'Argument with name {name} not found')