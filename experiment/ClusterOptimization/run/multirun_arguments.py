import math
from os import path
from typing import List, Tuple

import numpy as np

from analysis.utils.settings import ALEXNET_DIR
from experiment.utils.args import ExperimentArgParams
from experiment.utils.settings import FILE_NAMES
from pxdream.clustering.ds      import Clusters
from pxdream.utils.misc         import copy_exec
from pxdream.utils.parameters import ArgParams

def get_rnd_seed() -> str:
    return str(np.random.randint(1000, 100000))


ITER         = '150'
TEMPLATE     = 'T'
CLUSTER_ALGO = 'ds'
LAYER        = 'conv5-maxpool'
GEN_VARIANT  = 'fc8'
SAMPLE       = 5
A, B         = (500, 575)

NAME = f'cluster_single_units_{CLUSTER_ALGO}_from{A}_to_{B}_layer{LAYER}_{SAMPLE}samples_{ITER}iter'

LAYERS_SETTINGS = {    # Name           # Format               # Idx
    'fc8'           : ('fc8',           'alexnetfc8',          '21'),
    'fc7-relu'      : ('fc7-relu',      'alexnetfc7relu',      '20'),
    'fc7'           : ('fc7',           'alexnetfc7',          '19'),
    'fc6-relu'      : ('fc6-relu',      'alexnetfc6relu',      '17'),
    'conv5-maxpool' : ('conv5-maxpool', 'alexnetconv5maxpool', '13'),
}

LAYER_DIR, LAYER_FORMAT, LAYER_IDX = LAYERS_SETTINGS[LAYER]

CLU_DIR     = path.join(ALEXNET_DIR, LAYER_DIR)
CLU_FILE    = path.join(CLU_DIR, FILE_NAMES[CLUSTER_ALGO])
CLUSTERS    = Clusters.from_file(CLU_FILE)
CLUSTER_IDX = list(range(A, B+1))

def get_args_ds_weighting_scr() -> Tuple[str, str, str]:
    
    args = [
        (clu_idx, weighted, get_rnd_seed())
        for weighted in ['True', '']
        for clu_idx  in CLUSTER_IDX
        for _        in range(SAMPLE)
    ]
    
    clu_idx_str  = '#'.join([str(a) for a, _, _ in args])
    weighted_str = '#'.join([str(a) for _, a, _ in args])
    rnd_seed_str = '#'.join([str(a) for _, _, a in args])
    
    return clu_idx_str, weighted_str, rnd_seed_str

def get_args_cluster_superstimulus() -> Tuple[str, str]:
    
    args = [
        (clu_idx, get_rnd_seed())
        for clu_idx  in CLUSTER_IDX
        for _        in range(SAMPLE)
    ]
    
    clu_idx_str  = '#'.join([str(a) for a, _ in args])
    rnd_seed_str = '#'.join([str(a) for _, a in args])
    
    return clu_idx_str, rnd_seed_str

def get_args_cluster_units_superstimulus() -> Tuple[str, str, str]:
        
        args = [ 
                (opt_unit+1, clu_idx, get_rnd_seed())
                for clu_idx in CLUSTER_IDX 
                for opt_unit in range(len(CLUSTERS[clu_idx]))
                for _ in range(SAMPLE)
        ]
        
        opt_units_str = '#'.join([str(a) for a, _, _ in args])
        clu_idx_str   = '#'.join([str(a) for _, a, _ in args])
        rnd_seed_str  = '#'.join([str(a) for _, _, a in args])
        
        return clu_idx_str, opt_units_str, rnd_seed_str
    

def get_args_subsetting_optimization() -> Tuple[str, str, str]:
    
    args = [
        (clu_idx, topbot, k)
        for clu_idx in CLUSTER_IDX
        for k in [
            1, 
            int(math.sqrt(len(CLUSTERS[clu_idx]))),
            len(CLUSTERS[clu_idx])//2
        ]
        for topbot in ['subset_top', 'subset_bot']]
    
    clu_idx_str   = '#'.join([str(a) for a, _, _ in args])
    topbot_str    = '#'.join([str(a) for _, a, _ in args])
    opt_units_str = '#'.join([str(a) for _, _, a in args])
    
    return clu_idx_str, topbot_str, opt_units_str


if __name__ == '__main__':
    
    print('Multiple run: ')
    print('[1] DS Weighting ')
    print('[2] Cluster superstimulus ')
    print('[3] Cluster units supertimulus')
    print('[4] Cluster subsetting optimization')
    
    choice = int(input('Choice: '))
    
    match choice:
        
        case 1:
            
            assert CLUSTER_ALGO == 'ds', 'Invalid cluster algorithm for DS weighting'
            
            clu_idx_str, weighted_score_str, rnd_seed_str = get_args_ds_weighting_scr()
            
            args = {
                str(ExperimentArgParams.ClusterIdx)    : clu_idx_str,
                str(ExperimentArgParams.WeightedScore) : weighted_score_str,
                str(          ArgParams.RandomSeed)    : rnd_seed_str,
                str(ExperimentArgParams.ScoringType)   : 'cluster'
            }
            
            file = 'run_multiple_ds_weighting.py'
            
        case 2:
            
            clu_idx_str, rnd_seed_str = get_args_cluster_superstimulus()
            
            args = {
                str(ExperimentArgParams.ClusterIdx)    : clu_idx_str,
                str(          ArgParams.RandomSeed)    : rnd_seed_str,
                str(ExperimentArgParams.ScoringType)   : 'cluster'
            }
            
            file = 'run_multiple_cluster_superstimulus.py'
            
        case 3:
            
            clu_idx_str, opt_units_str, rnd_seed_str = get_args_cluster_units_superstimulus()
            
            args = {
                str(ExperimentArgParams.ClusterIdx)    : clu_idx_str,
                str(ExperimentArgParams.OptimUnits)    : opt_units_str,
                str(          ArgParams.RandomSeed)    : rnd_seed_str,
                str(ExperimentArgParams.ScoringType)   : 'subset'
            }
            
            file = 'run_multiple_cluster_units_superstimulus.py'
            
        case 4:
            
            clu_idx_str, topbot_str, opt_units_str = get_args_subsetting_optimization()
            
            args = {
                str(ExperimentArgParams.ClusterIdx)    : clu_idx_str,
                str(ExperimentArgParams.OptimUnits)    : opt_units_str,
                str(ExperimentArgParams.ScoringType)   : topbot_str,
            }
            
            file = 'run_multiple_cluster_subsetting_optimization.py'
            
        case 0:
            
            print('Exit')
            
        case _:
            raise ValueError('Invalid choice')
    
    
    args[str(          ArgParams.ExperimentName)] = NAME
    args[str(          ArgParams.NumIterations )] = ITER
    args[str(ExperimentArgParams.Template      )] = TEMPLATE
    args[str(ExperimentArgParams.GenVariant    )] = GEN_VARIANT
    args[str(ExperimentArgParams.ClusterLayer  )] = LAYER_IDX
    args[str(ExperimentArgParams.ClusterDir    )] = CLU_DIR
    args[str(ExperimentArgParams.ClusterAlgo   )] = CLUSTER_ALGO

    copy_exec(file=file, args=args)






