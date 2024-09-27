import os
from typing import Any, Dict, List, Type, cast
from experiment.utils.args import ExperimentArgParams
from experiment.utils.parsing import parse_boolean_string
from pxdream.experiment import Experiment, MultiExperiment, ZdreamExperiment
from pxdream.utils.parameters import ArgParam, ArgParams, ParamConfig, Parameter
from pxdream.utils.misc import overwrite_dict
from pxdream.utils.logger import DisplayScreen, Logger, LoguruLogger, SilentLogger
from IPython import get_ipython

def in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Eseguito in un Jupyter Notebook
        elif shell == 'TerminalInteractiveShell':
            return False  # Eseguito in un terminale IPython
        else:
            return False  # Altro ambiente
    except NameError:
        return False      # Non eseguito in IPython
# --- SINGLE RUN --- 

def param_from_str(name: str) -> ArgParam:
    try:               return           ArgParam.from_str(name)
    except ValueError: return ExperimentArgParams.from_str(name)

def convert_argparams_to_dict(argparams_dict):
    return {
        param.name: value
        for param, value in argparams_dict.items()
    }
def update_argparams(argparams_dict: Dict[ArgParam, Any], updates: Dict[str, Any]) -> Dict[ArgParam, Any]:
    for param in argparams_dict.keys():
        if param.name in updates:
            argparams_dict[param] = updates[param.name]
    return argparams_dict

def run_single(
    args_conf : ParamConfig,
    exp_type  : Type[Experiment],
    changes   : Dict[str, Any] = {}
):
    if in_notebook():
        cmd_conf = convert_argparams_to_dict(args_conf)
    else:
        # Parse cmd arguments
        parser   = ArgParams.get_parser(args=list(args_conf.keys()))
        cmd_conf = vars(parser.parse_args())
        cmd_conf = {
            param_from_str(arg_name) : value 
            for arg_name, value in cmd_conf.items() if value
        }

    # Merge configurations
    full_conf = overwrite_dict(args_conf, cmd_conf)
    full_conf = update_argparams(full_conf, changes)
    
    # Rendering
    if full_conf.get(ArgParams.Render, False):
        
        # Hold main display screen reference
        main_screen = DisplayScreen.set_main_screen()

        # Add close screen flag on as the experiment only involves one run
        full_conf[ArgParams.CloseScreen.value] = True
        
    experiment = exp_type.from_config(full_conf)
    experiment.run()
    

# --- MULTI RUN ---
    
def run_multi(
    args_conf      : ParamConfig,
    exp_type       : Type[Experiment],
    multi_exp_type : Type[MultiExperiment],
):       
    
    
    # Parse cmd arguments
    parser   = ArgParams.get_parser(args=list(args_conf.keys()), multirun=True)
    cmd_conf = vars(parser.parse_args())
    cmd_conf = {
        param_from_str(arg_name) : value 
        for arg_name, value in cmd_conf.items() if value
    }
    
    experiment = multi_exp_type.from_args(
        arg_conf     = cmd_conf,
        default_conf = args_conf,
        exp_type     = exp_type
    )
    
    experiment.run()
    
class BaseZdreamMultiExperiment(MultiExperiment):
    ''' Generic class handling different multi-experiment types. '''
    
    def __init__(
        self, 
        experiment:      Type['ZdreamExperiment'], 
        experiment_conf: Dict[ArgParam, List[Parameter]], 
        default_conf:    ParamConfig
    ) -> None:
        
        super().__init__(experiment, experiment_conf, default_conf)
        
        self._Exp: ZdreamExperiment = cast(ZdreamExperiment, self._Exp)
        
    @property
    def _logger_type(self) -> Type[Logger]: return LoguruLogger

    def _get_display_screens(self) -> List[DisplayScreen]:

        # Screen for synthetic images
        screens = [ 
            DisplayScreen(
                title=self._Exp.GEN_IMG_SCREEN,                 # type: ignore
                display_size=DisplayScreen.DEFAULT_DISPLAY_SIZE
            )
        ]

        # Add screen for natural images if at least one will use it
        use_nat = any(
            parse_boolean_string(str(conf[ExperimentArgParams.Template.value])).count(False) > 0 
            for conf in self._search_config
        )
        
        if use_nat:
            screens.append(
                DisplayScreen(
                    title=self._Exp.NAT_IMG_SCREEN,                 # type: ignore
                    display_size=DisplayScreen.DEFAULT_DISPLAY_SIZE
                )
            )

        return screens
    
# --- MISC --- 
    
def make_dir(path: str, logger: Logger = SilentLogger()) -> str:
    
    logger.info(f'Creating directory: {path}')
    os.makedirs(path, exist_ok=True)
    
    return path

