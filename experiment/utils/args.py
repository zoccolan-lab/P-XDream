from __future__ import annotations

from os import path
from enum import Enum

from zdream.utils.parameters import ArgParam
from zdream.utils.io_ import read_json

SCRIPT_DIR     = path.abspath(path.join(__file__, '..', '..'))
LOCAL_SETTINGS = path.join(SCRIPT_DIR, 'local_settings.json')

local_setting = read_json(LOCAL_SETTINGS)

OUT_DIR    = local_setting['out_dir']
WEIGHTS    = local_setting['weights']
DATASET    = local_setting['dataset']
IMAGE      = local_setting['image']
CLUSTERING = local_setting['clu_dir']

LAYERS_NEURONS_SPECIFICATION = '''
TODO write here how to specify neurons
'''

class ExperimentArgParams(Enum):
    
    # Generator
    GenWeights         = ArgParam(name="weights",            type=str,   help="Path to folder with generator weights")
    GenVariant         = ArgParam(name="variant",            type=str,   help="Variant of InverseAlexGenerator to use")
    
    # Natural image dataloader
    Dataset            = ArgParam(name="dataset",            type=str,   help="Path to mini-imagenet dataset")
    BatchSize          = ArgParam(name="batch_size",         type=int,   help="Natural image dataloader batch size")
    Template           = ArgParam(name="template",           type=str,   help="String of True(T) and False(F) as the basic sequence of the mask")
    Shuffle            = ArgParam(name="shuffle",            type=bool,  help="If to shuffle mask template")
    
    # Clustering
    ClusterDir         = ArgParam(name="clu_dir",            type=str,   help="Path to clustering directory")
    ClusterAlgo        = ArgParam(name="clu_algo",           type=str,   help="Name for clustering type {`gmm`, `nc`, `ds`}")
    MaxIterations      = ArgParam(name="max_iter",           type=int,   help="Maximum number of iterations")
    MinElements        = ArgParam(name="min_elements",       type=int,   help="Minimum cluster cardinality")
    ClusterIdx         = ArgParam(name="cluster_idx",        type=int,   help="Cluster index to optimize for")
    WeightedScore      = ArgParam(name="weighted_score",     type=bool,  help="If to weight score by cluster rank")
    ClusterLayer       = ArgParam(name="layer",              type=int,   help="Layer name for which clustering was computed")
    ScoringType        = ArgParam(name="scr_type",           type=str,   help="Scoring units strategy {`cluster`; `random`; `random_adj`, `subset_top`, `subset_bot`, `subset_rand`}")
    OptimUnits         = ArgParam(name="opt_units",          type=str,   help="Number of units to optimize in the cluster for `subset` scoring type")
    UseGPU             = ArgParam(name="gpu",                type=bool,  help="If to use GPU for clustering")
    NComponents        = ArgParam(name="n_components",       type=int,   help="Number of components to use for GMM clustering")
    NClusters          = ArgParam(name="n_clusters",         type=int,   help="Number of clusters to find for GMM and NC clustering")
    
    # Recording
    ImageIds           = ArgParam(name="image_ids",          type=str,   help="Image indexes for recording separated by a comma")
    LogCheckpoint      = ArgParam(name="log_chk",            type=int,   help="Logger iteration checkpoint")

    # Subject
    NetworkName        = ArgParam(name="net_name",           type=str,   help="SubjectNetwork name")
    RecordingLayers    = ArgParam(name="rec_layers",         type=str,   help=f"Recording layers with specification\n{LAYERS_NEURONS_SPECIFICATION}")

    # Scorer
    ScoringLayers      = ArgParam(name="scr_layers",         type=str,   help=f"Target scoring layers and neurons with specification\n{LAYERS_NEURONS_SPECIFICATION}")
    UnitsReduction     = ArgParam(name="units_reduction",    type=str,   help="Name of reducing function across units")
    LayerReduction     = ArgParam(name="layer_reduction",    type=str,   help="Name of reducing function across layers")

    # Optimizer
    OptimType          = ArgParam(name="optimizer_type",     type=str,   help="Type of optimizer. Either `genetic` or `cmaes`")
    PopulationSize     = ArgParam(name="pop_size",           type=int,   help="Starting number of the population")
    MutationRate       = ArgParam(name="mut_rate",           type=float, help="Mutation rate for the optimizer")
    MutationSize       = ArgParam(name="mut_size",           type=float, help="Mutation size for the optimizer")
    NumParents         = ArgParam(name="n_parents",          type=int,   help="Number of parents for the optimizer")
    TopK               = ArgParam(name="topk",               type=int,   help="Number of codes of previous generation to keep")
    Temperature        = ArgParam(name="temperature",        type=float, help="Temperature for the optimizer")
    TemperatureFactor  = ArgParam(name="temperature_factor", type=float, help="Temperature for the optimizer")
    RandomDistr        = ArgParam(name="random_distr",       type=str,   help="Random distribution for the codes initialization")
    AllowClones        = ArgParam(name="allow_clones",       type=str,   help="Random distribution for the codes initialization")
    RandomScale        = ArgParam(name="random_scale",       type=float, help="Random scale for the random distribution sampling")
    Sigma0             = ArgParam(name="sigma0",             type=float, help="Initial variance for CMAES covariance matrix")
    
    
    # --- MAGIC METHODS ---
    
    def __str__ (self)  -> str: return self.value.name
    def __repr__(self)  -> str: return str(self)
    
    
    @classmethod
    def from_str(cls, name: str) -> ArgParam:
        ''' Return the argument from the string name. '''
        
        for arg in cls:
            if str(arg) == name: return arg.value
        
        raise ValueError(f'Argument with name {name} not found')



