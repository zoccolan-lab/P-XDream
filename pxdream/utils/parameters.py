'''
This module contains the main classes implementing the argument parameters for the command line interface.
It implements the basic parameters to run an experiment, such as random seed, logging options, rendering, etc.
'''

from __future__ import annotations

from argparse import ArgumentParser
import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Type

from pxdream.utils.io_ import save_json
from pxdream.utils.parameters import ArgParam

Parameter      = str | int | float | bool
''' Possible command line argument types. '''

ParameterType  = Type[Parameter]
''' Type of the parameter. '''

ParamConfig  = Dict['ArgParam', Parameter]
''' Configuration with different possibles argument parameters. '''

@dataclass
class ArgParam:
    ''' Class to represent an argument parameter. '''

    name    : str
    ''' Name of the argument. '''

    help    : str
    ''' Description of the argument. '''

    type    : ParameterType
    ''' Type of the argument. '''

    default : Any = None
    '''
    Default value of the argument.
    None means that the argument is required with no default value.
    '''

    @classmethod
    def from_str(cls, name: str) -> ArgParam:
        ''' 
        Return the argument parameter from the string identifier.

        :param name: Parameter name identifier.
        :type name: str
        :return: Argument parameter matching with input string.
        :rtype: ArgParam
        '''
        
        for arg in ArgParams:
            if str(arg) == name: return arg.value
        
        raise ValueError(f'Argument with name {name} not found. Valid names are')
    
    # NOTE: The following methods are used to make the class hashable and a valid key in a dictionary
    def __hash__(self)        -> int:  return hash(self.name)
    def __eq__  (self, other) -> bool: return self.name == other.name if isinstance(other, ArgParam) else False
    
    @staticmethod
    def argconf_to_json(argconf: ParamConfig) -> Dict[str, Parameter]:
        '''
        Convert the argument configuration to JSON-like dictionary.
        
        :param argconf: Argument configuration to convert.
        :type argconf: ParamConfig
        :return: JSON-like dictionary with the argument configuration.
        :rtype: Dict[str, Parameter]
        '''
        
        return {arg.name : val for arg, val in argconf.items()}
    
    @staticmethod
    def save_argconf_to_json(argconf: ParamConfig, out_file: str = 'config.json') -> None:
        ''' 
        Save the argument configuration as a JSON file. 
        
        :param argconf: Argument configuration to save.
        :type argconf: ParamConfig
        '''

        argconf_json = ArgParam.argconf_to_json(argconf=argconf)
        
        save_json(data=argconf_json, path=out_file)




class ArgParams(Enum):
    '''
    Class to represent the different argument parameters for the command line interface.

    NOTE:   The main rational behind this class is to have a unique name and view of a 
            parameters which may be shared across different experiments. 
    '''

    # Logger
    ExperimentName     = ArgParam(name="name",          type=str,  help="Experiment name")
    ExperimentVersion  = ArgParam(name="version",       type=int,  help="Experiment version")
    OutputDirectory    = ArgParam(name="out_dir",       type=str,  help="Path to directory to save outputs")

    # Globals
    NumIterations      = ArgParam(name="iter",          type=int,  help="Number of total iterations")
    DisplayPlots       = ArgParam(name="display_plots", type=bool, help="If to display plots")
    RandomSeed         = ArgParam(name="random_seed",   type=int,  help="Random state for the experiment")
    Render             = ArgParam(name="render",        type=bool, help="If to render stimuli")
    
    # Extra

    # NOTE: The following are not proper arguments that are supposed to be passed from the user.
    #       They are used in an hidden way from the experiment run to handle 
    #        - the type of experiment used using the parameter `Title`.
    #        - the Screen buffer logic to share the same screen across multiple experiments in the case of a multi-run.
    
    ExperimentTitle    = ArgParam(name="title",           type=str,  help='')
    DisplayScreens     = ArgParam(name="display_screens", type=bool, help='') 
    CloseScreen        = ArgParam(name="close_screen",    type=bool, help='') 
    
    # --- MAGIC METHODS ---
    
    def __str__ (self)  -> str: return self.value.name
    def __repr__(self)  -> str: return str(self)
    
    # --- UTILITIES ---

    @staticmethod
    def str2bool(value: str) -> bool:
        ''' Helper function to deal with boolean input'''

        if isinstance(value, bool):                          return value
        if value.lower() in ('yes', 'true',  't', 'y', '1'): return True
        if value.lower() in ('no',  'false', 'f', 'n', '0'): return False
        raise argparse.ArgumentTypeError(f'Value `{value}` not recognized as a valid boolean value')
        
    @staticmethod
    def get_parser(
        args:     List[ArgParam] = [],
        multirun: bool      = False
    ) -> ArgumentParser:
        '''
        Produces an `ArgumentParser` given a list of input parameters.

        NOTE: In the case of `multirun` all parameters must have the same type `str` as they will specify
            multiple values for the same parameter interleaved with the separator `#`.
        '''

        parser = ArgumentParser()
        
        for arg in args:

            if arg.type != bool or multirun:
            
                parser.add_argument(
                    f"--{arg.name}", 
                    type=arg.type if not multirun else str, 
                    help=arg.help, 
                    default=arg.default
                )
            
            else:
                
                # For boolean parsing of a single experiment,
                # we use a more flexible str-to-bool conversion
                parser.add_argument(
                    f"--{arg.name}", 
                    type=ArgParams.str2bool, 
                    nargs='?',
                    const=True,
                    default=False,
                    help=arg.help
                )


        
        return parser
