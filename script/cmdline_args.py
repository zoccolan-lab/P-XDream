from argparse import ArgumentParser
from dataclasses import dataclass
from os import path
from typing import Any, Dict, List, Type
from dataclasses import dataclass
from typing import Any, Type
from enum import Enum

from zdream.utils.io_ import read_json, save_json

SCRIPT_DIR     = path.abspath(path.join(__file__, '..'))
LOCAL_SETTINGS = path.join(SCRIPT_DIR, 'local_settings.json')

local_setting = read_json(LOCAL_SETTINGS)

CONFIG     = local_setting['config']
OUT_DIR    = local_setting['out_dir']
WEIGHTS    = local_setting['weights']
DATASET    = local_setting['dataset']
IMAGE      = local_setting['image']
CLUSTERING = local_setting['clustering']
RECORDINGS = local_setting['recordings']

LAYERS_NEURONS_SPECIFICATION = '''
TODO write here how to specify neurons
'''

@dataclass
class Arg:
    
    name: str
    type: Type
    help: str
    default: Any = None

class Args(Enum):
    
    # Generator
    GenWeights   = Arg(name="weights", type=str, help="Path to folder with generator weights", default=WEIGHTS)
    GenVariant   = Arg(name="variant", type=str, help="Variant of InverseAlexGenerator to use")
    
    # Natural image dataloader
    Dataset   = Arg(name="dataset",    type=str,  help="Path to mini-imagenet dataset", default=DATASET)
    BatchSize = Arg(name="batch_size", type=int,  help="Natural image dataloader batch size")
    Template  = Arg(name="template",   type=str,  help="String of True(T) and False(F) as the basic sequence of the mask")
    Shuffle   = Arg(name="shuffle",    type=bool, help="If to shuffle mask template")
    
    # Clustering
    Recordings     = Arg(name="recordings",     type=str,  help="Path to neural recordings file", default=RECORDINGS)
    MaxIterations  = Arg(name="max_iter",       type=int,  help="Maximum number of iterations")
    MinElements    = Arg(name="min_elements",   type=int,  help="Minimum cluster cardinality")
    ClusterFile    = Arg(name="cluster_file",   type=str,  help="Path to clustering JSON file", default=CLUSTERING)
    ClusterIdx     = Arg(name="cluster_idx",    type=int,  help="Cluster index to optimize for")
    WeightedScore  = Arg(name="weighted_score", type=bool, help="If to weight score by cluster rank")
    ClusterLayer   = Arg(name="layer",          type=int,  help="Layer name for which clustering was computed")
    ScoringType    = Arg(name="scr_type",       type=str,  help="Scoring units strategy {`cluster`; `random`; `random_adj`, `subset_top`, `subset_bot`, `subset_rand`}")
    OptimUnits     = Arg(name="opt_units",      type=str,  help="Number of units to optimize in the cluster for `subset` scoring type")
    UseGPU         = Arg(name="gpu",            type=bool, help="If to use GPU for clustering")
    
    # Recording
    ImageIds      = Arg(name="image_ids",  type=str,  help="Image indexes for recording separated by a comma")
    LogCheckpoint = Arg(name="log_chk",    type=int,  help="Logger iteration checkpoint")

    # Subject
    NetworkName     = Arg(name="net_name",   type=str, help="SubjectNetwork name")
    RecordingLayers = Arg(name="rec_layers", type=str, help=f"Recording layers with specification\n{LAYERS_NEURONS_SPECIFICATION}")

    # Scorer
    ScoringLayers  = Arg(name="scr_layers",      type=str, help=f"Target scoring layers and neurons with specification\n{LAYERS_NEURONS_SPECIFICATION}")
    UnitsReduction = Arg(name="units_reduction", type=str, help="Name of reducing function across units")
    LayerReduction = Arg(name="layer_reduction", type=str, help="Name of reducing function across layers")

    # Optimizer
    OptimType          = Arg(name="optimizer_type",     type=str,   help="Type of optimizer. Either `genetic` or `cmaes`")
    PopulationSize     = Arg(name="pop_size",           type=int,   help="Starting number of the population")
    MutationRate       = Arg(name="mut_rate",           type=float, help="Mutation rate for the optimizer")
    MutationSize       = Arg(name="mut_size",           type=float, help="Mutation size for the optimizer")
    NumParents         = Arg(name="n_parents",          type=int,   help="Number of parents for the optimizer")
    TopK               = Arg(name="topk",               type=int,   help="Number of codes of previous generation to keep")
    Temperature        = Arg(name="temperature",        type=float, help="Temperature for the optimizer")
    TemperatureFactor  = Arg(name="temperature_factor", type=float, help="Temperature for the optimizer")
    RandomDistr        = Arg(name="random_distr",       type=str,   help="Random distribution for the codes initialization")
    AllowClones        = Arg(name="allow_clones",       type=str,   help="Random distribution for the codes initialization")
    RandomScale        = Arg(name="random_scale",       type=float, help="Random scale for the random distribution sampling")
    Sigma0             = Arg(name="sigma0",             type=float, help="Initial variance for CMAES covariance matrix")

    # Logger
    ExperimentName    = Arg(name="name",    type=str, help="Experiment name")
    ExperimentVersion = Arg(name="version", type=int, help="Experiment version")
    OutputDirectory   = Arg(name="out_dir", type=str, help="Path to directory to save outputs", default=OUT_DIR)

    # Globals
    NumIterations = Arg(name="iter",          type=int,  help="Number of total iterations")
    DisplayPlots  = Arg(name="display_plots", type=bool, help="If to display plots")
    RandomSeed    = Arg(name="random_seed",   type=int,  help="Random state for the experiment")
    Render        = Arg(name="render",        type=bool, help="If to render stimuli")
    
    # Additional
    ExperimentTitle = Arg(name="title",           type=str, help='')  # NOTE: Not a proper argument, depends on the experiment run
    DisplayScreens  = Arg(name="display_screens", type=int, help='')  # NOTE: Not a proper argument, key to pass a shared one-instance screen to multiple experiments
    
    def __str__ (self)  -> str: return self.value.name
    def __repr__(self)  -> str: return str(self)
    
    def __hash__(self): return hash(self.name)

    def __eq__(self, other):
        
        if isinstance(other, Args):
            return self.name == other.name
        return False
    
    @staticmethod
    def get_config_arg(conf_file: str) -> Arg:
        ''' Return the argument for the configuration file. '''
        
        return Arg(
            name="config", 
            type=str, 
            help="Path for the JSON configuration file", 
            default=path.join(CONFIG, conf_file)
        )
    
    @staticmethod
    def get_parser(
        args:     List[Arg] = [],
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
    
    @staticmethod
    def args_to_json(
        args : List[Arg] = [],
        fp   : str       = 'config.json'
    ) -> Dict[str, Any]:
        ''' Save and return a dictionary with the default values for the input arguments. '''
        
        DEFAULTS = {int: 0, float: 0., str: '', bool: False}
        
        json_dict = {arg.name: DEFAULTS[arg.type] for arg in args}        
        
        save_json(data=json_dict, path=fp)
        
        return json_dict
