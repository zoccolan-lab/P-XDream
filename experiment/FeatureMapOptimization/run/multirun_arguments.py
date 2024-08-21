import math
from os import path
import random
from typing import List, Tuple

import numpy as np

from experiment.utils.args import FEATURE_MAPS, ExperimentArgParams
from experiment.utils.settings import FILE_NAMES
from zdream.utils.io_ import read_json
from zdream.utils.misc         import copy_exec
from zdream.utils.parameters import ArgParams


NAME = 'prova'

ITER          = '3'
TEMPLATE      = 'T'
CLUSTER_ALGO  = 'ds'
SEG_TYPE      = 'clu'
SAMPLE        = 2

DATA = read_json(path.join(FEATURE_MAPS, f'{SEG_TYPE}_segmentation_optim.json'))[FILE_NAMES[CLUSTER_ALGO].replace('Clusters.json', '')]

FM_IDX = CLU_IDX = [i for i in range(50) if 3 <= len(DATA[str(i)]) <= 5][:2]

def get_rnd_seed() -> int: return int(random.randint(1000, 100000))

def get_args_fm_segments() -> Tuple[str, str, str]:
    
    match SEG_TYPE:
        
        case 'fm'  : idxs = FM_IDX
        case 'clu' : idxs = CLU_IDX
        case _     : raise ValueError(f"Invalid segmentation type {SEG_TYPE}. Must be in {{`fm`, `clu`}}")
    
    args = [
        (idx, k, get_rnd_seed())
        for idx in idxs
        for k   in DATA[str(idx)].keys()
        for _   in range(SAMPLE)
    ]
    
    idx_str      = '#'.join([str(a) for a, _, _ in args])
    fm_key_str   = '#'.join([str(a) for _, a, _ in args])
    rnd_seed_str = '#'.join([str(a) for _, _, a in args])
    
    return idx_str, fm_key_str, rnd_seed_str


if __name__ == '__main__':
    
    print('Multiple run: ')
    print('[1] FM Segmentation ')
    
    choice = int(input('Choice: '))
    
    match choice:
        
        case 1:
            
            idx_str, fm_key_str, rnd_seed_str = get_args_fm_segments()
            
            match SEG_TYPE:
        
                case 'fm'  : arg = ExperimentArgParams.FeatureMapIdx
                case 'clu' : arg = ExperimentArgParams.ClusterIdx
                case _     : raise ValueError(f"Invalid segmentation type {SEG_TYPE}. Must be in {{`fm`, `clu`}}")
            
            args = {
                str(arg)                            : idx_str, 
                str(ExperimentArgParams.FMKey     ) : fm_key_str,
                str(          ArgParams.RandomSeed) : rnd_seed_str,
            }
            
            file = 'run_multiple_fm_segment.py'
            
        case 0:
            
            print('Exit')
            
        case _:
            raise ValueError('Invalid choice')
    
    args[str(          ArgParams.ExperimentName    )] = NAME
    args[str(          ArgParams.NumIterations     )] = ITER
    args[str(ExperimentArgParams.Template          )] = TEMPLATE
    args[str(ExperimentArgParams.ClusterAlgo       )] = CLUSTER_ALGO
    args[str(ExperimentArgParams.FMSegmentationType)] = SEG_TYPE

    copy_exec(file=file, args=args)






